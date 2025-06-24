import sys, os, math, random, tempfile
import json 
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, \
    QHBoxLayout, QPushButton, QFileDialog, QLabel, QListWidget, QListWidgetItem, \
    QDoubleSpinBox, QFormLayout, QTabWidget, QDockWidget, QInputDialog, QAction, \
    QComboBox, QGroupBox, QCheckBox, QColorDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QVector3D, QQuaternion, QColor
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from ctypes import *
import pybullet as pb
try:
    import trimesh
except ImportError:
    trimesh = None
from PIL import Image
from PIL import Image
import numpy as np

FRAME_RATE = 60

# -----------------------
# SceneObject and MeshPart
# -----------------------
class MeshPart:
    def __init__(self):
        self.vertices = []
        self.normals = []
        self.uvs = None
        self.indices = []
        self.vertexCount = 0
        self.indexCount = 0
        self.Tangent = []
        self.Bitangent = []
        self.hasIndex = False
        self.texture_id = None
        self.normal_texture_id = None
    
    def constructTBN(self):
        if self.vertices and self.indices:
            verts = np.array(self.vertices, dtype=np.float32)
            n = len(verts)
            tangent_sum = np.zeros((n, 3), dtype=np.float32)
            bitangent_sum = np.zeros((n, 3), dtype=np.float32)
            count = np.zeros(n, dtype=np.int32)
            # Accumulate per-vertex tangents and bitangents
            for i in range(0, len(self.indices), 3):
                i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
                v0, v1, v2 = verts[i0], verts[i1], verts[i2]
                tangent = v1 - v0
                bitangent = v2 - v0
                tangent_sum[i0] += tangent; tangent_sum[i1] += tangent; tangent_sum[i2] += tangent
                bitangent_sum[i0] += bitangent; bitangent_sum[i1] += bitangent; bitangent_sum[i2] += bitangent
                count[i0] += 1; count[i1] += 1; count[i2] += 1
            # Average the tangents and bitangents
            for i in range(n):
                if count[i] > 0:
                    tangent_sum[i] /= count[i]
                    bitangent_sum[i] /= count[i]
            self.tangent = tangent_sum
            self.bitangent = bitangent_sum
            # Compute normals from cross product of tangent and bitangent
            self.normals = []
            for i in range(n):
                norm = np.cross(self.tangent[i], self.bitangent[i])
                self.normals.append([float(norm[0]), float(norm[1]), float(norm[2])])
            


class SceneObject:
    def __init__(self, name):
        self.name = name
        self.meshParts = []
        self.vertices = []  # list of QVector3D for bounding
        self.position = QVector3D(0.0, 0.0, 0.0)
        self.rotation = QQuaternion(1, 0, 0, 0)
        self.scale = QVector3D(1.0, 1.0, 1.0)
        self.shape_type = None
        self.mass = 1.0
        self.offset = QVector3D(0.0, 0.0, 0.0)
        self.radius = 0.0
        self.height = 0.0
        self.body_id = None
        self.inertia = QVector3D(0.0, 0.0, 0.0)
        self.show_mesh = True
        self.show_collision = False

class Light:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.color = (1.0, 1.0, 1.0)

# ------------------------
# GL Widget (OpenGL View)
# ------------------------
class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        # shader programs
        self.vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
in vec3 aTangent;
in vec3 aBitangent;
layout(location = 2) in vec2 aTexCoord;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;
out vec3 Tangent;
out vec3 Bitangent;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    TexCoord = aTexCoord; // Pass texture coordinates to fragment shader
    FragPos = vec3(model * vec4(aPos, 1.0)); // intermediate interpolated surface positions in 3D view space
    gl_Position = projection * view * vec4(FragPos, 1.0); // Transform the vertex position to clip space
    
    
    // Transform tangent, bitangent, and normal vectors to view space
    Tangent = normalize(mat3(model) * aTangent); 
    Bitangent = normalize(mat3(model) * aBitangent); 
    Normal = mat3(model) * normalize(aNormal);     

    
}
"""

        self.fragment_shader_source = """
#version 330 core
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec3 Tangent;
in vec3 Bitangent;

out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform sampler2D uTexture;
uniform sampler2D uNormalMap;

void main()
{
    // TBR matrix and tangent space calculations are not needed for this example
    mat3 TBN;
    TBN[0] = Tangent;
    TBN[1] = Bitangent;
    TBN[2] = Normal;

    

    // Sample normal map
    vec3 normalMap = texture(uNormalMap, TexCoord).rgb;

    // Convert normal map from [0,1] to [-1,1]
    vec3 normalTexel = normalize(normalMap * 2.0 - 1.0);
    
    // Transform and normalise normal from tangent space to world space
    vec3 normal = TBN * normalTexel;
    
    // Lighting calculations
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse 
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 lighting = ambient + diffuse + specular;

    // Sample texture 
    vec4 texColor = texture(uTexture, TexCoord);


    fragColor = texColor * vec4(lighting, 1.0);
}
"""     
        self.program_id = None

        self.setFocusPolicy(Qt.StrongFocus)
        # Scene objects and lights
        self.objects = []
        self.lights = []
        # Camera parameters
        self.camPos = QVector3D(0.0, 5.0, 15.0)
        self.camYaw = 0.0
        self.camPitch = 0.0
        # Movement
        self.keys = set()
        self.rotating = False
        self.last_mouse = None
        # Dragging physics objects
        self.drag_body = None
        self.drag_plane_y = 0.0
        # Selection marker size
        self.selected_object = None
        self.marker_size = 0.25
        # Physics (PyBullet)
        self.physics_client = None
        try:
            self.physics_client = pb.connect(pb.DIRECT)
            pb.setGravity(0, -9.81, 0)
            planeShape = pb.createCollisionShape(pb.GEOM_PLANE, planeNormal=[0,1,0])
            pb.createMultiBody(0, planeShape)
        except Exception as e:
            self.physics_client = None
            print("PyBullet not available:", e)
        # Sun icon texture
        self.sun_tex = 0
        # Placeholder for directional light
        self.sun_dir = [-0.5, -1.0, -0.5]
        # Start timer for update loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateScene)
        self.timer.start(int(1000 / FRAME_RATE))  # Limit redraw to FRAME_RATE FPS

    
    def compile_shader(self, source: str, shader_type: GLenum) -> GLuint:
        # 1) Create the shader object
        shader = glCreateShader(shader_type)
        if not shader:
            raise RuntimeError("glCreateShader failed")

        # 2) Set the shader source
        glShaderSource(shader, source)

        # 3) Compile it
        glCompileShader(shader)

        # 4) Check compile status
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if status == GL_FALSE:
            # Retrieve and print the error log
            log = glGetShaderInfoLog(shader).decode('utf‑8')
            stype = "VERTEX" if shader_type == GL_VERTEX_SHADER else "FRAGMENT"
            raise RuntimeError(f"{stype} shader compile error:\n{log}")

        return shader

    def create_program(self, vertex_src: str, fragment_src: str):
        # Compile each shader
        vert_shader = self.compile_shader(vertex_src,   GL_VERTEX_SHADER)
        frag_shader = self.compile_shader(fragment_src, GL_FRAGMENT_SHADER)

        # Create the program and attach shaders
        program = glCreateProgram()
        glAttachShader(program, vert_shader)
        glAttachShader(program, frag_shader)

        # Link the program
        glLinkProgram(program)

        # Check link status
        link_status = glGetProgramiv(program, GL_LINK_STATUS)
        if link_status == GL_FALSE:
            log = glGetProgramInfoLog(program).decode('utf‑8')
            raise RuntimeError(f"Program link error:\n{log}")

        # Once linked, you can delete the individual shaders
        glDeleteShader(vert_shader)
        glDeleteShader(frag_shader)

        return program

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        # disable all fixed‑function lighting—our shader handles it now
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 
        # shader setup
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        self.program_id = self.create_program(self.vertex_shader_source, self.fragment_shader_source)
        
        glUseProgram(self.program_id)
        
        # bind samplers to texture units once
        loc = glGetUniformLocation(self.program_id, "uTexture")
        if loc != -1: glUniform1i(loc, 0)
        loc = glGetUniformLocation(self.program_id, "uNormalMap")
        if loc != -1: glUniform1i(loc, 1)
        
        # cache uniform locations
        self.uModel    = glGetUniformLocation(self.program_id, "model")
        self.uView     = glGetUniformLocation(self.program_id, "view")
        self.uProj     = glGetUniformLocation(self.program_id, "projection")
        self.uLightPos = glGetUniformLocation(self.program_id, "lightPos")
        self.uViewPos  = glGetUniformLocation(self.program_id, "viewPos")
        self.uLightCol = glGetUniformLocation(self.program_id, "lightColor")
        
        # Load sun icon texture
        if os.path.exists("lightbulb.png"):
            try:
                img = Image.open("lightbb.png").convert("RGBA")
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                data = img.tobytes("raw", "RGBA", 0, -1)
                w, h = img.size
                self.sun_tex = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self.sun_tex)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, data)
                glGenerateMipmap(GL_TEXTURE_2D)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)
            except Exception as e:
                print("Failed to load sun texture:", e)

    def resizeGL(self, w, h):
        if h == 0: h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w/float(h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
# things that I need to do: create for each part multiple tex_id stored for each map. locate 
# the sampler variable identifier locaor, and bind the appropriate map to that 
# sampler variable. This will allow the shader to access the correct texture for each part.  
# breaking it apart even further I need to be able to sort the texture typ using a NAME FILTER
# Then I need to properly handle the lighting. to do this I need the 3d location of the texellb 
# being rastered at a specific point in 3d space (which typically the 2d projection space 
# coordinate of the texel is what the frag shader computes for us, which uses a geometric trick
# allowing it to bypass the need to compute the 3d location of the texel being rastered.) so 
# I need to compute the 3d location of the texel being rastered, to this end we need the 
# TBN matrix model. My code allready provides the Normal vector for the part class, but we need
# to compute the Tangent and Bitangent vectors for each part. To this end we need to compute
# the vectors that represent the edges of the triangle mesh primitive being rastered, for each
# triangle primitive seperately. vertexes that are shared between triangles will have to have 
# seperate unique Tangent and Bitangent vectors, as they are computed per triangle primitive.
# this presents a pragmatic challenge since it needs to.


# lets seperat the problem into a few sections:
# - Texture file name reading and filtering --> texture retrieval and correct ID.py (stored in barrel gltf folder)
# Texture binding and sampler variable location --> texture variables and shader modifications.dart in downloads
# computing the TBN matrix for each primitive --> Done inside of the vertex shader
# adjusting the shader to use the TBN matrix --> shader modifications.dart in downloads
# Rendering the scene with lighting and textures --> coompleted in the shader modifications.dart 


#now I need to calculate the normals and tangents and bi tangents for each part of the mesh.
# to do this i need to follow the following steps:
# For each primitive I need to compute the edges of the triangle mesh primitive.
# for each vertex I need to create an array for multiple tangent and bitangent vectors, which we will later average
# after that we will average the results. 
    def _makeProjectionMatrix(self, w, h):
        # returns a 4×4 perspective projection matrix matching gluPerspective
        fovy, aspect, near, far = 60.0, w/float(h), 0.1, 100.0
        f = 1.0 / math.tan(math.radians(fovy) / 2)
        M = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        return M.T.flatten()

    def _makeViewMatrix(self):
        # returns a 4×4 view matrix equivalent to gluLookAt
        yaw = math.radians(self.camYaw); pitch = math.radians(self.camPitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = np.array([cp*cy, sp, cp*sy], dtype=np.float32)
        eye = np.array([self.camPos.x(), self.camPos.y(), self.camPos.z()], dtype=np.float32)
        center = eye + fwd
        up = np.array([0,1,0], dtype=np.float32)
        # build look‑at basis
        z = eye - center; z /= np.linalg.norm(z)
        x = np.cross(up, z);  x /= np.linalg.norm(x)
        y = np.cross(z, x)
        M = np.eye(4, dtype=np.float32)
        M[0,:3], M[1,:3], M[2,:3] = x, y, z
        T = np.eye(4, dtype=np.float32); T[:3,3] = -eye
        return (M @ T).T.flatten()

    def _makeModelMatrix(self, pos, rot, scale):
        # translation
        T = np.eye(4, dtype=np.float32); T[:3,3] = [pos.x(), pos.y(), pos.z()]
        # scale
        S = np.diag([scale.x(), scale.y(), scale.z(), 1.0]).astype(np.float32)
        # rotation from quaternion
        w,x,y,z = rot.scalar(), rot.x(), rot.y(), rot.z()
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w,   0],
            [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w,   0],
            [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y, 0],
            [0,0,0,1]
        ], dtype=np.float32)
        M = T @ R @ S
        return M.T.flatten()
    
    def setup_mesh_part(self, part):
        # 1) Create a VAO to capture all state
        part.vao = glGenVertexArrays(1)
        glBindVertexArray(part.vao)

        # 2) Create & fill a single interleaved VBO
        #    [pos(xyz), normal(xyz), tangent(xyz), bitangent(xyz), uv(uv)] per‑vertex
        flat_data = []
        for i in range(part.vertexCount):
            flat_data.extend(part.vertices[i])
            flat_data.extend(part.normals[i])
            flat_data.extend(part.tangent[i])
            flat_data.extend(part.bitangent[i])
            flat_data.extend(part.uvs[i])

        # 2) Convert to a ctypes array of floats:
        ArrayType = c_float * len(flat_data)
        c_array = ArrayType(*flat_data)

        # 3) Upload to the VBO:
        part.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, part.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                    sizeof(c_array),            # total size in bytes
                    POINTER(c_float)(c_array),  # pointer to the first element
                    GL_STATIC_DRAW)

        # 3) If you have indices, create an EBO
        if part.hasIndex:
            idx_type = GLuint * len(part.indices)
            index_data = idx_type(*part.indices)
            part.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, part.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                        sizeof(index_data),
                        index_data,
                        GL_STATIC_DRAW)

        # 4) Configure vertex attributes (match your 'layout(location=…)'):
        stride = (3 + 3 + 3 + 3 + 2) * 4  # floats × 4 bytes

        # aPos @ location 0
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # aNormal @ location 1
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                            stride, ctypes.c_void_p(3 * 4))

        # aTangent @ location 3
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE,
                            stride, ctypes.c_void_p((3+3)*4))

        # aBitangent @ location 4
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE,
                            stride, ctypes.c_void_p((3+3+3)*4))

        # aTexCoord @ location 2
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                            stride, ctypes.c_void_p((3+3+3+3)*4))

        # Unbind VAO (optional but good practice)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        if part.hasIndex:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program_id)
        
        # upload camera uniforms each frame
        view = self._makeViewMatrix()         # implement to return 4×4 view matrix
        proj = self._makeProjectionMatrix(self.width(), self.height())   # implement to return 4×4 projection matrix
        glUniformMatrix4fv(self.uView,  1, GL_FALSE, view)
        glUniformMatrix4fv(self.uProj,  1, GL_FALSE, proj)
        glUniform3fv(self.uLightPos, 1, self.sun_dir)
        cam = [self.camPos.x(), self.camPos.y(), self.camPos.z()]
        glUniform3fv(self.uViewPos,  1, cam)
        glUniform3fv(self.uLightCol, 1, [1.0,1.0,1.0])
        
        # Draw ground grid
        self._draw_ground()
        # Draw sun icon billboard
        self._draw_sun()
        # Draw objects
        for obj in self.objects:
            if not obj.show_mesh:
                continue
            # build and upload model matrix
            model = self._makeModelMatrix(obj.position, obj.rotation, obj.scale)
            glUniformMatrix4fv(self.uModel, 1, GL_FALSE, model)
            
            for part in obj.meshParts:
                glUseProgram(self.program_id)
                glBindVertexArray(part.vao)

                # bind textures as you already do…
                glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, part.texture_id)
                glActiveTexture(GL_TEXTURE1);  glBindTexture(GL_TEXTURE_2D, part.normal_texture_id or 0)

                if part.hasIndex:
                    glDrawElements(GL_TRIANGLES, part.indexCount, GL_UNSIGNED_INT, None)
                else:
                    glDrawArrays(GL_TRIANGLES, 0, part.vertexCount)

                glBindVertexArray(0)
                
                
                
        # Draw selected object marker (simple red triangle)
        if self.selected_object is not None:
            obj = self.selected_object
            if obj.vertices:
                max_y = max((v.y() for v in obj.vertices), default=0.0)
                top_world_y = obj.position.y() + obj.scale.y() * max_y
                x0 = obj.position.x(); z0 = obj.position.z()
                size = self.marker_size
                glDisable(GL_LIGHTING)
                glColor3f(1.0, 0.0, 0.0)
                glBegin(GL_TRIANGLES)
                glVertex3f(x0, top_world_y + size, z0)
                glVertex3f(x0 - size, top_world_y, z0 - size)
                glVertex3f(x0 + size, top_world_y, z0 - size)
                glEnd()
                glEnable(GL_LIGHTING)

    def _draw_ground(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        s = 20
        y = 0.0
        for i in range(-s, s+1):
            glVertex3f(i, y, -s); glVertex3f(i, y, s)
            glVertex3f(-s, y, i); glVertex3f(s, y, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_sun(self):
        # Push attributes so sun drawing doesn’t affect scene state.
        glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT)
        # Disable lighting and blending for the sun icon
        glDisable(GL_LIGHTING)
        glDisable(GL_BLEND)
        # Ensure a white colour state.
        glColor3f(1.0, 1.0, 1.0)
        
        # Camera position for billboard positioning
        pos = self.camPos
        upv = QVector3D(0, 1, 0)
        forward = QVector3D(math.cos(math.radians(self.camYaw)),
                             math.sin(math.radians(self.camPitch)),
                             math.sin(math.radians(self.camYaw)))
        rightv = QVector3D.crossProduct(forward, upv)
        rightv.normalize()
        upv2 = QVector3D.crossProduct(rightv, forward)
        upv2.normalize()
        size = 2.0
        px, py, pz = pos.x(), pos.y(), pos.z()
        rx, ry, rz = rightv.x(), rightv.y(), rightv.z()
        ux, uy, uz = upv2.x(), upv2.y(), upv2.z()
        
        if self.sun_tex:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.sun_tex)
        else:
            # Fallback: draw a white quad so the light doesn’t tint yellow.
            glDisable(GL_TEXTURE_2D)
            glColor3f(1.0, 1.0, 1.0)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(px - rx*size - ux*size, py - ry*size - uy*size, pz - rz*size - uz*size)
        glTexCoord2f(1, 0); glVertex3f(px + rx*size - ux*size, py + ry*size - uy*size, pz + rz*size - uz*size)
        glTexCoord2f(1, 1); glVertex3f(px + rx*size + ux*size, py + ry*size + uy*size, pz + rz*size + uz*size)
        glTexCoord2f(0, 1); glVertex3f(px - rx*size + ux*size, py - ry*size + uy*size, pz - rz*size + uz*size)
        glEnd()
        
        if self.sun_tex:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
        
        glPopAttrib()
        # Re-enable lighting for subsequent drawing.
        glEnable(GL_LIGHTING)

    def updateScene(self):
        self._update_cam()
        if self.drag_body is not None and self.last_mouse:
            mx, my = self.last_mouse
            mv = glGetDoublev(GL_MODELVIEW_MATRIX)
            pj = glGetDoublev(GL_PROJECTION_MATRIX)
            vp = glGetIntegerv(GL_VIEWPORT)
            nx, ny = mx, vp[3] - my
            near = gluUnProject(nx, ny, 0.0, mv, pj, vp)
            far = gluUnProject(nx, ny, 1.0, mv, pj, vp)
            if near and far:
                n = QVector3D(near[0], near[1], near[2])
                f = QVector3D(far[0], far[1], far[2])
                rd = QVector3D(f.x()-n.x(), f.y()-n.y(), f.z()-n.z())
                rd_norm = math.sqrt(rd.x()*rd.x() + rd.y()*rd.y() + rd.z()*rd.z())
                if rd_norm > 1e-6:
                    rd = QVector3D(rd.x()/rd_norm, rd.y()/rd_norm, rd.z()/rd_norm)
                    if abs(rd.y()) > 1e-6:
                        t = (self.drag_plane_y - n.y()) / rd.y()
                        newp = QVector3D(n.x() + rd.x()*t, self.drag_plane_y, n.z() + rd.z()*t)
                        pb.resetBasePositionAndOrientation(self.drag_body, [newp.x(), newp.y(), newp.z()], [0,0,0,1])
        self.update()

    def _update_cam(self):
        s = 0.3
        if Qt.Key_W in self.keys:
            ry = math.radians(self.camYaw)
            self.camPos.setX(self.camPos.x() + s*math.cos(ry))
            self.camPos.setZ(self.camPos.z() - s*math.sin(ry))
        if Qt.Key_S in self.keys:
            ry = math.radians(self.camYaw)
            self.camPos.setX(self.camPos.x() - s*math.cos(ry))
            self.camPos.setZ(self.camPos.z() + s*math.sin(ry))
        if Qt.Key_D in self.keys:
            ry = math.radians(self.camYaw - 90)
            self.camPos.setX(self.camPos.x() + s*math.cos(ry))
            self.camPos.setZ(self.camPos.z() - s*math.sin(ry))
        if Qt.Key_A in self.keys:
            ry = math.radians(self.camYaw + 90)
            self.camPos.setX(self.camPos.x() + s*math.cos(ry))
            self.camPos.setZ(self.camPos.z() - s*math.sin(ry))
        if Qt.Key_Q in self.keys:
            self.camPos.setY(self.camPos.y() + s)
        if Qt.Key_E in self.keys:
            self.camPos.setY(self.camPos.y() - s)

    def keyPressEvent(self, e):
        self.keys.add(e.key())

    def keyReleaseEvent(self, e):
        if e.key() in self.keys:
            self.keys.remove(e.key())

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.rotating = True
            self.last_mouse = (e.x(), e.y())
        elif e.button() == Qt.LeftButton and self.objects:
            self.drag_body = self.objects[0].body_id
            if self.drag_body is not None:
                pos, _ = pb.getBasePositionAndOrientation(self.drag_body)
                self.drag_plane_y = pos[1]
            self.last_mouse = (e.x(), e.y())

    def mouseMoveEvent(self, e):
        if self.rotating and self.last_mouse:
            dx = e.x() - self.last_mouse[0]
            dy = e.y() - self.last_mouse[1]
            self.camYaw += dx * 0.2
            self.camPitch -= dy * 0.2
            self.camPitch = max(-89.0, min(89.0, self.camPitch))
            self.last_mouse = (e.x(), e.y())
        else:
            self.last_mouse = (e.x(), e.y())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self.rotating = False
        if e.button() == Qt.LeftButton and self.drag_body is not None:
            pb.resetBaseVelocity(self.drag_body, [0,0,0], [0,0,0])
            self.drag_body = None

    def addObjectFromFile(self, file_path):
        if trimesh is None:
            print("trimesh not available; cannot import model.")
            return None
        base_color_imgs = []
        normal_imgs = []
        # Handle both .gltf and .glb files for base-color extraction
        if file_path.lower().endswith((".gltf", ".glb")):
            try:
                if file_path.lower().endswith(".gltf"):
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    textures = data.get("textures", [])
                    images = data.get("images", [])
                    materials = data.get("materials", [])
                    base_color_imgs = []
                    normal_imgs = []
                    for material in materials:
                        pbr = material.get("pbrMetallicRoughness", {})
                        # Base color texture
                        if "baseColorTexture" in pbr:
                            tex_index = pbr["baseColorTexture"].get("index")
                            if tex_index is not None and tex_index < len(textures):
                                img_index = textures[tex_index].get("source")
                                if img_index is not None and img_index < len(images):
                                    uri = images[img_index].get("uri", "")
                                    if uri and not uri.startswith("data:"):
                                        img_path = os.path.join(os.path.dirname(file_path), uri)
                                        try:
                                            pil_img = Image.open(img_path).convert("RGBA")
                                            base_color_imgs.append(pil_img)
                                            print("Loaded base color from:", img_path)
                                        except Exception as e:
                                            print("Failed to load base color image:", img_path, e)
                        # Normal texture
                        if "normalTexture" in material:
                            tex_index = material["normalTexture"].get("index")
                            if tex_index is not None and tex_index < len(textures):
                                img_index = textures[tex_index].get("source")
                                if img_index is not None and img_index < len(images):
                                    uri = images[img_index].get("uri", "")
                                    if uri and not uri.startswith("data:"):
                                        img_path = os.path.join(os.path.dirname(file_path), uri)
                                        try:
                                            pil_img = Image.open(img_path).convert("RGBA")
                                            normal_imgs.append(pil_img)
                                            print("Loaded normal map from:", img_path)
                                        except Exception as e:
                                            print("Failed to load normal image:", img_path, e)
            except Exception as e:
                 print("Failed to extract base color images:", e)
        
        print(len(base_color_imgs))
        
        try:
            scene = trimesh.load(file_path, force='scene')
        except Exception as e:
            print("Failed to load model:", e)
            return None
        name = os.path.splitext(os.path.basename(file_path))[0]
        obj = SceneObject(name)
        geoms = scene.dump() if hasattr(scene, 'dump') else [scene]
        print(geoms)
        for idx, geom in enumerate(geoms):
            if hasattr(geom, 'faces') and geom.faces is not None and len(geom.faces):
                verts = geom.vertices.copy()
                norms = geom.vertex_normals.copy() if geom.vertex_normals is not None else None
                faces = geom.faces.copy()
                part = MeshPart()
                part.vertices = verts.tolist()
                #part.normals = norms.tolist() if norms is not None else None
                uvs = None
                if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                    uvs = geom.visual.uv.copy()
                    part.uvs = uvs.tolist()
                part.indices = faces.flatten().tolist()
                part.indexCount = len(part.indices)
                part.vertexCount = len(part.vertices)
                part.constructTBN()  # Calculate TBN vectors for the part
                part.hasIndex = True
                part.texture_id = None
                if base_color_imgs and part.uvs:
                    try:
                        img = base_color_imgs[0]
                        data = img.tobytes("raw","RGBA",0,-1)       
                        w, h = img.size
                        tex_id = glGenTextures(1)
                        glBindTexture(GL_TEXTURE_2D, tex_id)
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                        glGenerateMipmap(GL_TEXTURE_2D)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                        glBindTexture(GL_TEXTURE_2D, 0)
                        part.texture_id = tex_id
                    except Exception as e:
                        print("Texture creation failed:", e)
                        part.texture_id = None
                # Load normal map if available
                part.normal_texture_id = None
                if normal_imgs and part.uvs:
                    try:
                        img = normal_imgs[0]
                        data = img.tobytes("raw","RGBA",0,-1)
                        w, h = img.size
                        tex_id2 = glGenTextures(1)
                        glBindTexture(GL_TEXTURE_2D, tex_id2)
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                        glGenerateMipmap(GL_TEXTURE_2D)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                        glBindTexture(GL_TEXTURE_2D, 0)
                        part.normal_texture_id = tex_id2
                    except Exception as e:
                        print("Normal texture creation failed:", e)
                        part.normal_texture_id = None
                obj.meshParts.append(part)
                for v in verts:
                    obj.vertices.append(QVector3D(v[0], v[1], v[2]))
        self.objects.append(obj)
        self.setup_mesh_part(part)
        print("added object to parts list")
        return obj

    def assignPhysicsShape(self, obj, shape_type):
        if self.physics_client is None or shape_type is None:
            return
        obj.shape_type = shape_type

        # Build a flattened list of scaled vertices
        vert_list = []
        for v in obj.vertices:
            vx = v.x() * obj.scale.x()
            vy = v.y() * obj.scale.y()
            vz = v.z() * obj.scale.z()
            vert_list.extend([vx, vy, vz])
        if not vert_list:
            return

        if shape_type == "Convex Hull":
            # Build a single list of triangle indices across all mesh parts,
            # with correct offsets for each part's vertex numbering.
            idx_list = []
            offset = 0
            
            if len(vert_list) // 3 > 50:
                vertex_count = len(vert_list) // 3
                # Randomly sample 150 indices
                keep_indices = sorted(random.sample(range(vertex_count), 50))
                # Rebuild the simplified vertex list
                simplified = []
                for i in keep_indices:
                    simplified.extend(vert_list[i*3:i*3+3])
                vert_list = simplified
            
            if len(vert_list) // 3 < 4:
                print("Convex hull requires at least 4 vertices, found:", len(vert_list) // 3)
                return
            
            for part in obj.meshParts:
                # part.indices already flattened as [i0,i1,i2, ...]
                idx_list.extend([idx + offset for idx in part.indices])
                offset += part.vertexCount
            
            mesh = trimesh.Trimesh(vertices=np.array(vert_list).reshape(-1,3), faces=[])
            tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
            mesh.export(tmp.name)
            try:
                print(len(vert_list))
                shape_id = pb.createCollisionShape(pb.GEOM_MESH, fileName=tmp.name)
            
                # shape_id = pb.createCollisionShape(
                #     pb.GEOM_MESH,
                #     vertices=vert_list,
                #     indices=idx_list,
                #     meshScale=[1,1,1]
                # )
            except Exception as e:
                print("Convex hull creation failed:", e)
                return
            
            obj.radius = 0.0
            obj.height = 0.0
            obj.offset = QVector3D(0, 0, 0)
            
    
        elif shape_type == "Sphere":
            # Compute radius as the maximum absolute extent in any direction
            rx = max(abs(v) for v in vert_list[0::3])
            ry = max(abs(v) for v in vert_list[1::3])
            rz = max(abs(v) for v in vert_list[2::3])
            radius = max(rx, ry, rz)
            shape_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
            obj.radius = radius
            # Set offset to 0 so that the collision shape aligns with object position
            obj.offset = QVector3D(0, 0, 0)
            obj.height = 0.0
        elif shape_type == "Cylinder":
            # Use X and Z for radius, and Y range for height
            rx = max(abs(v) for v in vert_list[0::3])
            rz = max(abs(v) for v in vert_list[2::3])
            radius = max(rx, rz)
            y_values = vert_list[1::3]
            y_min = min(y_values)
            y_max = max(y_values)
            height = y_max - y_min
            shape_id = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=radius, height=height)
            obj.radius = radius
            obj.height = height
            # Set offset so the collision shape’s center is at the object’s center
            obj.offset = QVector3D(0, (y_max + y_min) / 2.0, 0)
        else:
            return

        body_id = pb.createMultiBody(obj.mass, shape_id, -1,
                                     [obj.position.x(), obj.position.y(), obj.position.z()],
                                     [obj.rotation.x(), obj.rotation.y(), obj.rotation.z(), obj.rotation.scalar()],
                                     [obj.offset.x(), obj.offset.y(), obj.offset.z()])
        obj.body_id = body_id
        if body_id is not None:
            info = pb.getDynamicsInfo(body_id, -1)
            if info:
                ix, iy, iz = info[2]
                obj.inertia = QVector3D(ix, iy, iz)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.currentObject = None
        self.currentLight = None
        self.setWindowTitle("PyQt OpenGL Scene Editor")
        # Central widget
        self.glWidget = GLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.glWidget.setFocus()
        # Objects dock
        self.objectList = QListWidget()
        self.objectList.setSelectionMode(QListWidget.SingleSelection)
        objectDock = QDockWidget("Objects", self)
        objectDock.setWidget(self.objectList)
        objectDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, objectDock)
        # Lights dock
        self.lightList = QListWidget()
        self.lightList.setSelectionMode(QListWidget.SingleSelection)
        lightDock = QDockWidget("Lights", self)
        lightDock.setWidget(self.lightList)
        lightDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, lightDock)
        # Properties dock with tabs
        propDock = QDockWidget("Properties", self)
        propDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        tabs = QTabWidget()
        # Transform tab
        transWidget = QWidget()
        transForm = QFormLayout(transWidget)
        self.posXSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.posXSpin, -10000,10000,0.1)
        self.posYSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.posYSpin, -10000,10000,0.1)
        self.posZSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.posZSpin, -10000,10000,0.1)
        posLayout = QHBoxLayout(); posLayout.addWidget(self.posXSpin); posLayout.addWidget(self.posYSpin); posLayout.addWidget(self.posZSpin)
        posContainer = QWidget(); posContainer.setLayout(posLayout)
        transForm.addRow("Position:", posContainer)
        self.rotWSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.rotWSpin, -1,1,0.001)
        self.rotXSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.rotXSpin, -1,1,0.001)
        self.rotYSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.rotYSpin, -1,1,0.001)
        self.rotZSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.rotZSpin, -1,1,0.001)
        rotLayout = QHBoxLayout(); rotLayout.addWidget(self.rotWSpin); rotLayout.addWidget(self.rotXSpin); rotLayout.addWidget(self.rotYSpin); rotLayout.addWidget(self.rotZSpin)
        rotContainer = QWidget(); rotContainer.setLayout(rotLayout)
        transForm.addRow("Rotation (quat):", rotContainer)
        self.scaleXSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.scaleXSpin, 0.001,1000,0.01); self.scaleXSpin.setValue(1.0)
        self.scaleYSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.scaleYSpin, 0.001,1000,0.01); self.scaleYSpin.setValue(1.0)
        self.scaleZSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.scaleZSpin, 0.001,1000,0.01); self.scaleZSpin.setValue(1.0)
        scaleLayout = QHBoxLayout(); scaleLayout.addWidget(self.scaleXSpin); scaleLayout.addWidget(self.scaleYSpin); scaleLayout.addWidget(self.scaleZSpin)
        scaleContainer = QWidget(); scaleContainer.setLayout(scaleLayout)
        transForm.addRow("Scale:", scaleContainer)
        transForm.addRow("", QPushButton("Apply", clicked=self.applyTransformEdits))
        tabs.addTab(transWidget, "Transform")
        # Physics tab
        physWidget = QWidget()
        physForm = QFormLayout(physWidget)
        self.shapeCombo = QComboBox(); self.shapeCombo.addItems(["None", "Convex Hull", "Sphere", "Cylinder"])
        physForm.addRow("Shape:", self.shapeCombo)
        self.massSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.massSpin, 0,10000,0.1); self.massSpin.setValue(1.0)
        physForm.addRow("Mass:", self.massSpin)
        self.offsetXSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.offsetXSpin, -1000,1000,0.01)
        self.offsetYSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.offsetYSpin, -1000,1000,0.01)
        self.offsetZSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.offsetZSpin, -1000,1000,0.01)
        offsetLayout = QHBoxLayout(); offsetLayout.addWidget(self.offsetXSpin); offsetLayout.addWidget(self.offsetYSpin); offsetLayout.addWidget(self.offsetZSpin)
        offsetContainer = QWidget(); offsetContainer.setLayout(offsetLayout)
        physForm.addRow("Offset:", offsetContainer)
        self.radiusSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.radiusSpin, 0,10000,0.01)
        physForm.addRow("Radius:", self.radiusSpin)
        self.heightSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.heightSpin, 0,10000,0.01)
        physForm.addRow("Height:", self.heightSpin)
        self.inertiaXSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.inertiaXSpin, 0,100000,0.1)
        self.inertiaYSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.inertiaYSpin, 0,100000,0.1)
        self.inertiaZSpin = QDoubleSpinBox(); self._setupDoubleSpin(self.inertiaZSpin, 0,100000,0.1)
        inertiaLayout = QHBoxLayout(); inertiaLayout.addWidget(self.inertiaXSpin); inertiaLayout.addWidget(self.inertiaYSpin); inertiaLayout.addWidget(self.inertiaZSpin)
        inertiaContainer = QWidget(); inertiaContainer.setLayout(inertiaLayout)
        physForm.addRow("Inertia:", inertiaContainer)
        tabs.addTab(physWidget, "Physics")
        # Appearance tab
        appWidget = QWidget()
        appForm = QFormLayout(appWidget)
        self.colorDialog = QPushButton("Change Color", clicked=self.changeColor)
        appForm.addRow("Color:", self.colorDialog)
        self.showMeshCheckbox = QCheckBox()
        self.showMeshCheckbox.setChecked(True)
        self.showMeshCheckbox.stateChanged.connect(self.toggleMeshVisibility)
        appForm.addRow("Show Mesh:", self.showMeshCheckbox)
        self.showCollCheckbox = QCheckBox()
        self.showCollCheckbox.stateChanged.connect(self.toggleCollisionVisibility)
        appForm.addRow("Show Collision:", self.showCollCheckbox)
        tabs.addTab(appWidget, "Appearance")
        # Lighting tab
        lightWidget = QWidget()
        lightForm = QFormLayout(lightWidget)
        self.sunX = QDoubleSpinBox(); self._setupDoubleSpin(self.sunX, -10,10,0.1); self.sunX.setValue(self.glWidget.sun_dir[0])
        self.sunY = QDoubleSpinBox(); self._setupDoubleSpin(self.sunY, -10,10,0.1); self.sunY.setValue(self.glWidget.sun_dir[1])
        self.sunZ = QDoubleSpinBox(); self._setupDoubleSpin(self.sunZ, -10,10,0.1); self.sunZ.setValue(self.glWidget.sun_dir[2])
        lightForm.addRow("Sun Dir X,Y,Z:", self._hbox(self.sunX, self.sunY, self.sunZ))
        lightForm.addRow("", QPushButton("Update Lights", clicked=self.updateLights))
        tabs.addTab(lightWidget, "Lighting")
        propDock.setWidget(tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, propDock)

        # Menu for importing
        importAction = QAction("Import 3D Model", self, triggered=self.importModel)
        menu = self.menuBar().addMenu("File")
        menu.addAction(importAction)

        # Signals
        self.objectList.itemSelectionChanged.connect(self.onObjectSelected)
        self.lightList.itemSelectionChanged.connect(self.onLightSelected)
        self.posXSpin.editingFinished.connect(self.applyTransformEdits)
        self.posYSpin.editingFinished.connect(self.applyTransformEdits)
        self.posZSpin.editingFinished.connect(self.applyTransformEdits)
        self.rotWSpin.editingFinished.connect(self.applyTransformEdits)
        self.rotXSpin.editingFinished.connect(self.applyTransformEdits)
        self.rotYSpin.editingFinished.connect(self.applyTransformEdits)
        self.rotZSpin.editingFinished.connect(self.applyTransformEdits)
        self.scaleXSpin.editingFinished.connect(self.applyTransformEdits)
        self.scaleYSpin.editingFinished.connect(self.applyTransformEdits)
        self.scaleZSpin.editingFinished.connect(self.applyTransformEdits)
        self.shapeCombo.currentIndexChanged.connect(self.onShapeComboChanged)
        self.massSpin.editingFinished.connect(self.onPhysicsParamEdited)
        self.offsetXSpin.editingFinished.connect(self.onPhysicsParamEdited)
        self.offsetYSpin.editingFinished.connect(self.onPhysicsParamEdited)
        self.offsetZSpin.editingFinished.connect(self.onPhysicsParamEdited)
        self.radiusSpin.editingFinished.connect(self.onPhysicsParamEdited)
        self.heightSpin.editingFinished.connect(self.onPhysicsParamEdited)

        self.changeColor()
        self.updatePropertyFields()

    def _setupDoubleSpin(self, spin, minimum, maximum, step):
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)

    def _hbox(self, *args):
        layout = QHBoxLayout()
        for a in args:
            layout.addWidget(a)
        container = QWidget()
        container.setLayout(layout)
        return container

    def importModel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Import 3D Model", "",
                                                  "3D Model Files (*.gltf *.glb);;All Files (*)", options=options)
        if fileName:
            obj = self.glWidget.addObjectFromFile(fileName)
            if obj:
                item = QListWidgetItem(obj.name)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.objectList.addItem(item)
                self.glWidget.selected_object = obj
                self.onObjectSelected()

    def onObjectSelected(self):
        items = self.objectList.selectedItems()
        if not items:
            self.currentObject = None
            return
        idx = self.objectList.row(items[0])
        self.currentObject = self.glWidget.objects[idx]
        self.updatePropertyFields()

    def onLightSelected(self):
        items = self.lightList.selectedItems()
        if not items:
            self.currentLight = None
            return
        idx = self.lightList.row(items[0])
        self.currentLight = self.glWidget.lights[idx]
        self.updatePropertyFields()

    def updatePropertyFields(self):
        obj = self.currentObject
        if obj:
            self.posXSpin.setValue(obj.position.x()); self.posYSpin.setValue(obj.position.y()); self.posZSpin.setValue(obj.position.z())
            self.rotWSpin.setValue(obj.rotation.scalar()); self.rotXSpin.setValue(obj.rotation.x()); 
            self.rotYSpin.setValue(obj.rotation.y()); self.rotZSpin.setValue(obj.rotation.z())
            self.scaleXSpin.setValue(obj.scale.x()); self.scaleYSpin.setValue(obj.scale.y()); self.scaleZSpin.setValue(obj.scale.z())
            # Physics
            self.shapeCombo.setCurrentText(obj.shape_type if obj.shape_type else "None")
            self.massSpin.setValue(obj.mass)
            self.offsetXSpin.setValue(obj.offset.x()); self.offsetYSpin.setValue(obj.offset.y()); self.offsetZSpin.setValue(obj.offset.z())
            self.radiusSpin.setValue(obj.radius); self.heightSpin.setValue(obj.height)
            self.inertiaXSpin.setValue(obj.inertia.x()); self.inertiaYSpin.setValue(obj.inertia.y()); self.inertiaZSpin.setValue(obj.inertia.z())
            self.showMeshCheckbox.setChecked(obj.show_mesh)
            self.showCollCheckbox.setChecked(obj.show_collision)
        light = self.currentLight
        if light:
            self.sunX.setValue(self.glWidget.sun_dir[0]); self.sunY.setValue(self.glWidget.sun_dir[1]); self.sunZ.setValue(self.glWidget.sun_dir[2])

    def applyTransformEdits(self):
        if not self.currentObject: return
        obj = self.currentObject
        obj.position = QVector3D(self.posXSpin.value(), self.posYSpin.value(), self.posZSpin.value())
        q = QQuaternion(self.rotWSpin.value(), self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value())
        if q.length() == 0:
            q = QQuaternion(1.0, 0.0, 0.0, 0.0)
        else:
            q.normalize()
        obj.rotation = q
        self.rotWSpin.setValue(obj.rotation.scalar()); self.rotXSpin.setValue(obj.rotation.x());
        self.rotYSpin.setValue(obj.rotation.y()); self.rotZSpin.setValue(obj.rotation.z())
        obj.scale = QVector3D(self.scaleXSpin.value(), self.scaleYSpin.value(), self.scaleZSpin.value())
        if obj.body_id is not None:
            pos = [obj.position.x(), obj.position.y(), obj.position.z()]
            orn = [obj.rotation.x(), obj.rotation.y(), obj.rotation.z(), obj.rotation.scalar()]
            pb.resetBasePositionAndOrientation(obj.body_id, pos, orn)
            self.rebuildPhysicsBody(obj)
            if obj.body_id is not None:
                info = pb.getDynamicsInfo(obj.body_id, -1)
                if info:
                    ix, iy, iz = info[2]
                    obj.inertia = QVector3D(ix, iy, iz)
                    self.inertiaXSpin.setValue(ix); self.inertiaYSpin.setValue(iy); self.inertiaZSpin.setValue(iz)
        self.glWidget.update()

    def applyPhysics(self):
        if not self.currentObject: return
        obj = self.currentObject
        obj.mass = self.massSpin.value()
        shape = self.shapeCombo.currentText()
        if shape and shape != "None":
            if obj.body_id is not None:
                pb.removeBody(obj.body_id)
            obj.shape_type = shape
            self.glWidget.assignPhysicsShape(obj, shape)
        self.updatePropertyFields()

    def rebuildPhysicsBody(self, obj):
        if obj.body_id is not None:
            pb.removeBody(obj.body_id)
        if obj.shape_type:
            self.glWidget.assignPhysicsShape(obj, obj.shape_type)

    def onShapeComboChanged(self, index):
        if not self.currentObject: return
        shapeText = self.shapeCombo.currentText()
        obj = self.currentObject
        if shapeText == "None":
            if obj.body_id is not None:
                pb.removeBody(obj.body_id)
            obj.body_id = None
            obj.shape_type = None
        else:
            if obj.body_id is not None:
                pb.removeBody(obj.body_id)
            obj.shape_type = shapeText
            self.glWidget.assignPhysicsShape(obj, shapeText)
        if obj.body_id is not None:
            info = pb.getDynamicsInfo(obj.body_id, -1)
            if info:
                ix, iy, iz = info[2]
                obj.inertia = QVector3D(ix, iy, iz)
        self.inertiaXSpin.setValue(obj.inertia.x()); self.inertiaYSpin.setValue(obj.inertia.y()); self.inertiaZSpin.setValue(obj.inertia.z())

    def onPhysicsParamEdited(self):
        if not self.currentObject: 
            return
        obj = self.currentObject
        obj.offset = QVector3D(self.offsetXSpin.value(), self.offsetYSpin.value(), self.offsetZSpin.value())
        obj.radius = self.radiusSpin.value()
        obj.height = self.heightSpin.value()
        if obj.body_id is not None:
            self.rebuildPhysicsBody(obj)
            if obj.body_id is not None:
                info = pb.getDynamicsInfo(obj.body_id, -1)
                if info:
                    ix, iy, iz = info[2]
                    obj.inertia = QVector3D(ix, iy, iz)
        self.updatePropertyFields()

    def updateLights(self):
        self.glWidget.sun_dir = [self.sunX.value(), self.sunY.value(), self.sunZ.value()]
        self.glWidget.update()

    def toggleMeshVisibility(self, state):
        if self.currentObject:
            self.currentObject.show_mesh = (state == Qt.Checked)
            self.glWidget.update()

    def toggleCollisionVisibility(self, state):
        if self.currentObject:
            self.currentObject.show_collision = (state == Qt.Checked)
            self.glWidget.update()

    def changeColor(self):
        if not self.currentObject: 
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.currentObject.color = (color.redF(), color.greenF(), color.blueF())
            self.glWidget.update()

# Shader source code

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

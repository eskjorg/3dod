# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Renders rgb/depth image of a 3D mesh model.

import numpy as np
from glumpy import app, gloo, gl

# Set backend (http://glumpy.readthedocs.io/en/latest/api/app-backends.html)
app.use('glfw')
# app.use('qt5')
# app.use('pyside')

# Set logging level
from glumpy.log import log
import logging
log.setLevel(logging.WARNING) # ERROR, WARNING, DEBUG, INFO

# Color vertex shader
#-------------------------------------------------------------------------------
_color_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_nm;
uniform mat4 u_mvp;
uniform vec3 u_light_eye_pos;

in vec3 a_position;
in vec3 a_normal;
in vec3 a_color;
in vec2 a_texcoord;

out vec3 v_color;
out vec2 v_texcoord;
out vec3 v_eye_pos;
out vec3 v_L;
out vec3 v_normal;
out float v_eye_depth;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
    v_texcoord = a_texcoord;
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coords.

    // OpenGL Z axis goes out of the screen, so depths are negative
    v_eye_depth = -v_eye_pos.z;

    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light
    v_normal = normalize(u_nm * vec4(a_normal, 1.0)).xyz; // Normal in eye coords.
}
"""

# Color fragment shader - Phong shading
#-------------------------------------------------------------------------------
_color_fragment_phong_code = """
uniform float u_light_ambient_w;
uniform sampler2D u_texture;
uniform int u_use_texture;

in vec3 v_color;
in vec2 v_texcoord;
in vec3 v_eye_pos;
in vec3 v_L;
in vec3 v_normal;
in float v_eye_depth;

layout(location = 0) out vec4 cb_rgb;
layout(location = 1) out vec4 cb_depth;

void main() {
    float light_diffuse_w = max(dot(normalize(v_L), normalize(v_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;

    if(bool(u_use_texture))
        cb_rgb = vec4(light_w * texture2D(u_texture, v_texcoord));
    else
        cb_rgb = vec4(light_w * v_color, 1.0);
    cb_depth = vec4(v_eye_depth, 0.0, 0.0, 1.0);
}
"""

# Functions to calculate transformation matrices
# Note that OpenGL expects the matrices to be saved column-wise
# (Ref: http://www.songho.ca/opengl/gl_transform.html)
#-------------------------------------------------------------------------------
# Model-view matrix
def _compute_model_view(model, view):
    return np.dot(model, view)

# Model-view-projection matrix
def _compute_model_view_proj(model, view, proj):
    return np.dot(np.dot(model, view), proj)

# Normal matrix (Ref: http://www.songho.ca/opengl/gl_normaltransform.html)
def _compute_normal_matrix(model, view):
    return np.linalg.inv(np.dot(model, view)).T

# Conversion of Hartley-Zisserman intrinsic matrix to OpenGL projection matrix
#-------------------------------------------------------------------------------
# Ref:
# 1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
# 2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
def _compute_calib_proj(K, x0, y0, w, h, nc, fc, window_coords='y_down'):
    """
    :param K: Camera matrix.
    :param x0, y0: The camera image origin (normally (0, 0)).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: OpenGL projection matrix.
    """
    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same
    if window_coords == 'y_up':
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ]) # This row is also standard glPerspective

    # Draw the images right side up and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords
    else:
        assert window_coords == 'y_down'
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ]) # This row is also standard glPerspective
    return proj.T

#-------------------------------------------------------------------------------
def draw(shape, vertex_buffer, index_buffer, texture, mat_model, mat_view,
               mat_proj, ambient_weight, bg_color):

    color_fragment_code = _color_fragment_phong_code

    program = gloo.Program(_color_vertex_code, color_fragment_code, version='330')
    program.bind(vertex_buffer)
    program['u_light_eye_pos'] = [0, 0, 0] # Camera origin
    program['u_light_ambient_w'] = ambient_weight
    program['u_mv'] = _compute_model_view(mat_model, mat_view)
    program['u_nm'] = _compute_normal_matrix(mat_model, mat_view)
    program['u_mvp'] = _compute_model_view_proj(mat_model, mat_view, mat_proj)
    if texture is not None:
        program['u_use_texture'] = int(True)
        program['u_texture'] = texture
    else:
        program['u_use_texture'] = int(False)
        program['u_texture'] = np.zeros((1, 1, 4), np.float32)

    # Frame buffer object
    color_buf_rgb = np.zeros((shape[0], shape[1], 4), np.float32).view(gloo.TextureFloat2D)
    color_buf_depth = np.zeros((shape[0], shape[1], 4), np.float32).view(gloo.TextureFloat2D)
    depth_buf = np.zeros((shape[0], shape[1]), np.float32).view(gloo.DepthTexture)
    fbo = gloo.FrameBuffer(
        color = [
            color_buf_rgb,
            color_buf_depth,
        ],
        depth = depth_buf,
    )
    fbo.activate()

    # OpenGL setup
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glViewport(0, 0, shape[1], shape[0])

    # gl.glEnable(gl.GL_BLEND)
    # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    # gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
    # gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
    # gl.glDisable(gl.GL_LINE_SMOOTH)
    # gl.glDisable(gl.GL_POLYGON_SMOOTH)
    # gl.glEnable(gl.GL_MULTISAMPLE)

    # Keep the back-face culling disabled because of objects which do not have
    # well-defined surface (e.g. the lamp from the dataset of Hinterstoisser)
    gl.glDisable(gl.GL_CULL_FACE)
    # gl.glEnable(gl.GL_CULL_FACE)
    # gl.glCullFace(gl.GL_BACK) # Back-facing polygons will be culled

    # Rendering
    program.draw(gl.GL_TRIANGLES, index_buffer)

    # Retrieve the contents of the FBO texture

    rgb = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_FLOAT, rgb)
    rgb.shape = shape[0], shape[1], 4
    rgb = rgb[::-1, :]
    rgb = np.round(rgb[:, :, :3] * 255).astype(np.uint8) # Convert to [0, 255]

    depth = np.zeros((shape[0], shape[1]), dtype=np.float32)
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
    gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RED, gl.GL_FLOAT, depth)
    depth = depth[::-1, :]
    # depth = depth[:, :, 0] # Depth is saved in the first channel

    fbo.deactivate()

    return rgb, depth

#-------------------------------------------------------------------------------
def render(model, im_size, K, R, t, clip_near=100, clip_far=2000,
           texture=None, surf_color=None, bg_color=(0.0, 0.0, 0.0, 0.0),
           ambient_weight=0.5):

    # Process input data
    #---------------------------------------------------------------------------
    # Make sure vertices and faces are provided in the model
    assert({'pts', 'faces'}.issubset(set(model.keys())))

    # Set texture / color of vertices
    if texture is not None:
        if texture.max() > 1.0:
            texture = texture.astype(np.float32) / 255.0
        texture = np.flipud(texture)
        texture_uv = model['texture_uv']
        colors = np.zeros((model['pts'].shape[0], 3), np.float32)
    else:
        texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)
        if not surf_color:
            if 'colors' in model.keys():
                assert(model['pts'].shape[0] == model['colors'].shape[0])
                colors = model['colors']
                if colors.max() > 1.0:
                    colors /= 255.0 # Color values are expected in range [0, 1]
            else:
                colors = np.ones((model['pts'].shape[0], 3), np.float32) * 0.5
        else:
            colors = np.tile(list(surf_color) + [1.0], [model['pts'].shape[0], 1])

    # Set the vertex data
    vertices_type = [('a_position', np.float32, 3),
                     ('a_normal', np.float32, 3),
                     ('a_color', np.float32, colors.shape[1]),
                     ('a_texcoord', np.float32, 2)]
    vertices = np.array(list(zip(model['pts'], model['normals'],
                            colors, texture_uv)), vertices_type)

    # Model matrix
    mat_model = np.eye(4, dtype=np.float32) # From object space to world space

    # View matrix (transforming also the coordinate system from OpenCV to
    # OpenGL camera space)
    mat_view = np.eye(4, dtype=np.float32) # From world space to eye space
    mat_view[:3, :3], mat_view[:3, 3] = R, t.squeeze()
    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    mat_view = yz_flip.dot(mat_view) # OpenCV to OpenGL camera system
    mat_view = mat_view.T # OpenGL expects column-wise matrix format

    # Projection matrix
    mat_proj = _compute_calib_proj(K, 0, 0, im_size[0], im_size[1], clip_near, clip_far)

    # Create buffers
    vertex_buffer = vertices.view(gloo.VertexBuffer)
    index_buffer = model['faces'].flatten().astype(np.uint32).view(gloo.IndexBuffer)

    # Create window
    # config = app.configuration.Configuration()
    # Number of samples used around the current pixel for multisample
    # anti-aliasing (max is 8)
    # config.samples = 8
    # config.profile = "core"
    # window = app.Window(config=config, visible=False)
    window = app.Window(visible=False)

    global rgb, depth
    rgb = None
    depth = None

    @window.event
    def on_draw(dt):
        window.clear()
        shape = (im_size[1], im_size[0])
        global rgb
        global depth
        rgb, depth = draw(shape, vertex_buffer, index_buffer, texture, mat_model,
                         mat_view, mat_proj, ambient_weight, bg_color)

    app.run(framecount=0) # The on_draw function is called framecount+1 times
    window.close()

    return rgb, depth

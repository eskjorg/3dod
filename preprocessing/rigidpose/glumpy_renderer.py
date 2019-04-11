# Python / glumpy OpenGL renderer.
# Inspiration & pieces of code taken from https://github.com/thodan/sixd_toolkit/blob/master/pysixd/renderer.py

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

class Renderer():
    _vertex_shader = """
    uniform mat4 u_mv;
    uniform mat4 u_nm;
    uniform mat4 u_mvp;
    uniform vec3 u_light_eye_pos;

    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_color;
    in vec2 in_texcoord;

    out vec3 vs_color;
    out vec2 vs_texcoord;
    out vec3 vs_eye_pos;
    out vec3 vs_light_eye_dir;
    out vec3 vs_normal;
    out float vs_eye_depth;

    void main() {
        gl_Position = u_mvp * vec4(in_position, 1.0);
        vs_color = in_color;
        vs_texcoord = in_texcoord;
        vs_eye_pos = (u_mv * vec4(in_position, 1.0)).xyz; // Vertex position in eye frame.

        // OpenGL Z axis goes out of the screen, so depths are negative
        vs_eye_depth = -vs_eye_pos.z;

        vs_light_eye_dir = normalize(u_light_eye_pos - vs_eye_pos); // Vector to the light
        vs_normal = normalize(u_nm * vec4(in_normal, 1.0)).xyz; // Normal in eye frame.
    }
    """

    _fragment_shader = """
    uniform float u_light_ambient_w;
    uniform sampler2D u_texture_map;
    uniform int u_use_texture;

    in vec3 vs_color;
    in vec2 vs_texcoord;
    in vec3 vs_eye_pos;
    in vec3 vs_light_eye_dir;
    in vec3 vs_normal;
    in float vs_eye_depth;

    layout(location = 0) out vec4 out_rgb;
    layout(location = 1) out vec4 out_depth;

    void main() {
        float light_diffuse_w = max(dot(normalize(vs_light_eye_dir), normalize(vs_normal)), 0.0);
        float light_w = u_light_ambient_w + light_diffuse_w;
        if(light_w > 1.0) light_w = 1.0;

        if(bool(u_use_texture))
            out_rgb = vec4(light_w * texture2D(u_texture_map, vs_texcoord));
        else
            out_rgb = vec4(light_w * vs_color, 1.0);
        out_depth = vec4(vs_eye_depth, 0.0, 0.0, 1.0);
    }
    """

    def __init__(self, shape, K, clip_near = 100, clip_far = 10000):
        self._shape = shape
        self._mat_view = self._get_model_view_transf()
        self._mat_proj = self._compute_calib_proj(K, 0, 0, self._shape[1], self._shape[0], clip_near, clip_far)

    def _get_model_view_transf(self):
        # View matrix (transforming also the coordinate system from OpenCV to
        # OpenGL camera frame)
        mat_view = np.eye(4, dtype=np.float32)
        mat_view[1, 1], mat_view[2, 2] = -1, -1
        mat_view = mat_view.T # OpenGL expects column-wise matrix format
        return mat_view

    # Functions to calculate transformation matrices
    # Note that OpenGL expects the matrices to be saved column-wise
    # (Ref: http://www.songho.ca/opengl/gl_transform.html)
    #-------------------------------------------------------------------------------
    # Model-view matrix
    def _compute_model_view(self, model, view):
        return np.dot(model, view)

    # Model-view-projection matrix
    def _compute_model_view_proj(self, model, view, proj):
        return np.dot(np.dot(model, view), proj)

    # Normal matrix (Ref: http://www.songho.ca/opengl/gl_normaltransform.html)
    def _compute_normal_matrix(self, model, view):
        return np.linalg.inv(np.dot(model, view)).T

    # Conversion of Hartley-Zisserman intrinsic matrix to OpenGL projection matrix
    #-------------------------------------------------------------------------------
    # Ref:
    # 1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    # 2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
    def _compute_calib_proj(self, K, x0, y0, w, h, nc, fc, window_coords='y_down'):
        """
        :param K: Camera calibration matrix.
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
    def _setup_program(self, color_vertex_shader, color_fragment_code, ambient_weight, glsl_version='330'):
        program = gloo.Program(color_vertex_shader, color_fragment_code, version=glsl_version)
        program['u_light_eye_pos'] = [0, 0, 0] # Camera origin
        program['u_light_ambient_w'] = ambient_weight
        return program

    #-------------------------------------------------------------------------------
    def _create_framebuffer(self):

        # Frame buffer object
        color_buf_rgb = np.zeros((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        color_buf_depth = np.zeros((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        depth_buf = np.zeros((self._shape[0], self._shape[1]), np.float32).view(gloo.DepthTexture)
        fbo = gloo.FrameBuffer(
            color = [
                color_buf_rgb,
                color_buf_depth,
            ],
            depth = depth_buf,
        )
        fbo.activate()

        return fbo

    #-------------------------------------------------------------------------------
    def _prepare_rendering(self, bg_color):

        # OpenGL setup
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self._shape[1], self._shape[0])

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

    def _draw(self, program, R, t, obj_id, model, texture_map = None, surf_color = None):
        # Process input data
        #---------------------------------------------------------------------------
        # Make sure vertices and faces are provided in the model
        assert({'pts', 'faces'}.issubset(set(model.keys())))

        # Set texture / color of vertices
        if texture_map is not None:
            if texture_map.max() > 1.0:
                texture_map = texture_map.astype(np.float32) / 255.0
            texture_map = np.flipud(texture_map)
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
        vertices_type = [('in_position', np.float32, 3),
                         ('in_normal', np.float32, 3),
                         ('in_color', np.float32, colors.shape[1]),
                         ('in_texcoord', np.float32, 2)]
        vertices = np.array(list(zip(model['pts'], model['normals'],
                                colors, texture_uv)), vertices_type)

        # Model matrix
        mat_model = np.eye(4, dtype=np.float32) # From world frame to eye frame
        mat_model[:3, :3], mat_model[:3, 3] = R, t.squeeze()
        mat_model = mat_model.T

        # Create buffers
        vertex_buffer = vertices.view(gloo.VertexBuffer)
        index_buffer = model['faces'].flatten().astype(np.uint32).view(gloo.IndexBuffer)

        # Rendering
        program['u_mv'] = self._compute_model_view(mat_model, self._mat_view)
        program['u_nm'] = self._compute_normal_matrix(mat_model, self._mat_view)
        program['u_mvp'] = self._compute_model_view_proj(mat_model, self._mat_view, self._mat_proj)
        if texture_map is not None:
            program['u_use_texture'] = int(True)
            program['u_texture_map'] = texture_map
        else:
            program['u_use_texture'] = int(False)
            program['u_texture_map'] = np.zeros((1, 1, 4), np.float32)
        program.bind(vertex_buffer)
        program.draw(gl.GL_TRIANGLES, index_buffer)

    def _read_fbo(self):
        rgb = np.zeros((self._shape[0], self._shape[1], 4), dtype=np.float32)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RGBA, gl.GL_FLOAT, rgb)
        rgb = np.flipud(rgb)
        # rgb.shape = self._shape[0], self._shape[1], 4
        # rgb = rgb[::-1, :]
        rgb = np.round(rgb[:, :, :3] * 255).astype(np.uint8) # Convert to [0, 255]

        depth = np.zeros((self._shape[0], self._shape[1]), dtype=np.float32)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
        gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RED, gl.GL_FLOAT, depth)
        depth = np.flipud(depth)
        # depth = depth[::-1, :]
        # depth = depth[:, :, 0] # Depth is saved in the first channel

        return rgb, depth

    #-------------------------------------------------------------------------------
    def render(
        self,
        model_list,
        R_list,
        t_list,
        obj_id_list,
        texture_map_list = None,
        surf_color_list = None,
        bg_color = (0.0, 0.0, 0.0, 0.0),
        ambient_weight = 0.5,
    ):
        nbr_instances = len(model_list)
        if texture_map_list is None:
            texture_map_list = [None] * nbr_instances
        if surf_color_list is None:
            surf_color_list = [None] * nbr_instances

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

        program = self._setup_program(self._vertex_shader, self._fragment_shader, ambient_weight)
        fbo = self._create_framebuffer()
        self._prepare_rendering(bg_color)

        @window.event
        def on_draw(dt):
            window.clear()
            for model, R, t, obj_id, texture_map, surf_color in zip(model_list, R_list, t_list, obj_id_list, texture_map_list, surf_color_list):
                self._draw(program, R, t, obj_id, model, texture_map = texture_map, surf_color = surf_color)

        app.run(framecount=0) # The on_draw function is called framecount+1 times
        rgb, depth = self._read_fbo()

        fbo.deactivate()

        window.close()

        return rgb, depth
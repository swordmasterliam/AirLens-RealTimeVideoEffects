"""Unit tests for GLSL shaders, compilation, hot-reload, and error handling."""

import os
import unittest
import glfw
from OpenGL.GL import *
from shader_manager import ShaderManager


class TestShaderManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.VISIBLE, False)
        cls.window = glfw.create_window(100, 100, "ShaderTest", None, None)
        glfw.make_context_current(cls.window)

    @classmethod
    def tearDownClass(cls):
        if cls.window:
            glfw.destroy_window(cls.window)
        glfw.terminate()

    def setUp(self):
        shaders_dir = os.path.join(os.path.dirname(__file__), "..", "shaders")
        self.sm = ShaderManager(shaders_dir)

    def tearDown(self):
        self.sm.cleanup()

    def test_all_shaders_loaded(self):
        self.assertGreaterEqual(self.sm.shader_count, 10)
        self.assertIsNone(self.sm.last_error)

    def test_cycle_shaders(self):
        initial = self.sm.active_index
        self.sm.next_shader()
        self.assertEqual(self.sm.active_index, (initial + 1) % self.sm.shader_count)
        self.sm.prev_shader()
        self.assertEqual(self.sm.active_index, initial)

    def test_shader_compilation_error_graceful(self):
        """Verify that malformed GLSL does not crash the engine and records error."""
        bad_frag = """#version 330 core
        void main() {
            this is invalid glsl code !!!;
        }
        """
        from shader_manager import VERTEX_SHADER_SOURCE
        bad_prog = self.sm._compile_shader_from_source("bad", VERTEX_SHADER_SOURCE, bad_frag, "")
        self.assertIsNone(bad_prog)
        self.assertIsNotNone(self.sm.last_error)
        self.assertIn("Frag Shader Error", self.sm.last_error)

    def test_render_all_shaders_gpu(self):
        """Render a full quad with each shader on the GPU and check for GL errors."""
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        for i in range(self.sm.shader_count):
            self.sm.select_shader(i)
            for mode in (0, 1, 2):
                self.sm.render_quad(
                    texture_id=tex,
                    width=640,
                    height=480,
                    elapsed_time=1.23,
                    hand_box=(0.2, 0.2, 0.8, 0.8),
                    has_box=1.0,
                    mask_mode=mode,
                    intensity=1.0,
                    mouse_uv=(0.5, 0.5),
                    thumb1=(0.2, 0.2), index1=(0.3, 0.2),
                    thumb2=(0.8, 0.8), index2=(0.7, 0.8)
                )
                err = glGetError()
                self.assertEqual(err, GL_NO_ERROR, f"GL error {err} in shader {self.sm.active_shader_name} mode {mode}")

        glDeleteTextures(1, [tex])

    def test_hot_reload_disk(self):
        """Verify modifying a shader file triggers hot-reload."""
        temp_shader = os.path.join(self.sm.shaders_dir, "99_test_hotreload.frag")
        initial_count = self.sm.shader_count
        try:
            with open(temp_shader, "w", encoding="utf-8") as f:
                f.write("""#version 330 core
                out vec4 FragColor;
                in vec2 v_uv;
                uniform sampler2D u_camera;
                void main() { FragColor = texture(u_camera, v_uv); }
                """)

            reloaded = self.sm.check_hot_reload()
            self.assertTrue(reloaded)
            self.assertEqual(self.sm.shader_count, initial_count + 1)
        finally:
            if os.path.exists(temp_shader):
                os.remove(temp_shader)
            self.sm.check_hot_reload()

    def test_include_resolution_common_glsl(self):
        """Verify common.glsl resolution and inclusion."""
        frag_with_include = """#version 330 core
        out vec4 FragColor;
        in vec2 v_uv;
        uniform sampler2D u_camera;
        uniform vec4 u_hand_box;
        uniform float u_has_box;
        uniform int u_mask_mode;

        #include "common.glsl"

        void main() {
            float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);
            FragColor = texture(u_camera, v_uv) * mask;
        }
        """
        from shader_manager import VERTEX_SHADER_SOURCE
        resolved = self.sm._resolve_includes(frag_with_include, self.sm.shaders_dir)
        self.assertIn("compute_mask", resolved)
        prog = self.sm._compile_shader_from_source("test_include", VERTEX_SHADER_SOURCE, resolved, "")
        self.assertIsNotNone(prog)
        if prog:
            prog.delete()

    def test_four_finger_polygon_bounds_mask_mode_0(self):
        """Verify shader mask is bounded inside the 4-finger polygon rather than axis-aligned bbox."""
        import numpy as np

        frag_src = """#version 330 core
        out vec4 FragColor;
        in vec2 v_uv;
        uniform vec4 u_hand_box;
        uniform float u_has_box;
        uniform int u_mask_mode;

        #include "common.glsl"

        void main() {
            float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);
            FragColor = vec4(mask, mask, mask, 1.0);
        }
        """
        from shader_manager import VERTEX_SHADER_SOURCE
        resolved = self.sm._resolve_includes(frag_src, self.sm.shaders_dir)
        prog = self.sm._compile_shader_from_source("test_poly_mask", VERTEX_SHADER_SOURCE, resolved, "")
        self.assertIsNotNone(prog)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 100, 100, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glViewport(0, 0, 100, 100)

        prog.use()
        # Diamond quadrilateral: left=(0.2, 0.5), top=(0.5, 0.2), right=(0.8, 0.5), bottom=(0.5, 0.8)
        # Bounding box is (0.2, 0.2, 0.8, 0.8)
        prog.set_uniform_4f("u_hand_box", 0.2, 0.2, 0.8, 0.8)
        prog.set_uniform_1f("u_has_box", 1.0)
        prog.set_uniform_1i("u_mask_mode", 0)
        prog.set_uniform_2f("u_thumb1", 0.2, 0.5)
        prog.set_uniform_2f("u_index1", 0.5, 0.2)
        prog.set_uniform_2f("u_thumb2", 0.8, 0.5)
        prog.set_uniform_2f("u_index2", 0.5, 0.8)

        glBindVertexArray(self.sm.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        data = glReadPixels(0, 0, 100, 100, GL_RGBA, GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape((100, 100, 4))

        # Center UV (0.5, 0.5) is inside the diamond polygon -> should have full mask (255)
        self.assertEqual(img[50, 50, 0], 255)
        # Top-left corner of bounding box UV (0.25, 0.25) is inside bbox but outside polygon -> should have 0 mask
        self.assertEqual(img[75, 25, 0], 0)
        # Outside bbox UV (0.05, 0.05) -> should have 0 mask
        self.assertEqual(img[95, 5, 0], 0)

        # Mode 1: inverted (inside diamond is 0, outside is 255)
        prog.set_uniform_1i("u_mask_mode", 1)
        glBindVertexArray(self.sm.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        data_inv = glReadPixels(0, 0, 100, 100, GL_RGBA, GL_UNSIGNED_BYTE)
        img_inv = np.frombuffer(data_inv, dtype=np.uint8).reshape((100, 100, 4))
        self.assertEqual(img_inv[50, 50, 0], 0)
        self.assertEqual(img_inv[75, 25, 0], 255)

        # Fallback to box when 2 fingers (1 hand): thumb2 and index2 are (0, 0)
        prog.set_uniform_1i("u_mask_mode", 0)
        prog.set_uniform_2f("u_thumb2", 0.0, 0.0)
        prog.set_uniform_2f("u_index2", 0.0, 0.0)
        glBindVertexArray(self.sm.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        data_box = glReadPixels(0, 0, 100, 100, GL_RGBA, GL_UNSIGNED_BYTE)
        img_box = np.frombuffer(data_box, dtype=np.uint8).reshape((100, 100, 4))
        # Now corner inside bbox UV (0.25, 0.25) is inside the box -> 255
        self.assertEqual(img_box[75, 25, 0], 255)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [tex])
        prog.delete()

    def test_quad_mask_no_horizontal_spike_glitch(self):
        """Verify quad mask does not leak a horizontal triangular spike to the left of the hands (Image 1 & Image 2 bug)."""
        import numpy as np

        frag_src = """#version 330 core
        out vec4 FragColor;
        in vec2 v_uv;
        uniform vec4 u_hand_box;
        uniform float u_has_box;
        uniform int u_mask_mode;

        #include "common.glsl"

        void main() {
            float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);
            FragColor = vec4(mask, mask, mask, 1.0);
        }
        """
        from shader_manager import VERTEX_SHADER_SOURCE
        resolved = self.sm._resolve_includes(frag_src, self.sm.shaders_dir)
        prog = self.sm._compile_shader_from_source("test_spike_fix", VERTEX_SHADER_SOURCE, resolved, "")
        self.assertIsNotNone(prog)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 100, 100, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glViewport(0, 0, 100, 100)

        prog.use()
        # Realistic coordinates from Image 2:
        # Left hand: thumb1=(0.39, 0.54), index1=(0.39, 0.45)
        # Right hand: thumb2=(0.64, 0.68), index2=(0.63, 0.23)
        prog.set_uniform_4f("u_hand_box", 0.39, 0.23, 0.64, 0.68)
        prog.set_uniform_1f("u_has_box", 1.0)
        prog.set_uniform_1i("u_mask_mode", 0)
        prog.set_uniform_2f("u_thumb1", 0.39, 0.54)
        prog.set_uniform_2f("u_index1", 0.39, 0.45)
        prog.set_uniform_2f("u_thumb2", 0.64, 0.68)
        prog.set_uniform_2f("u_index2", 0.63, 0.23)

        glBindVertexArray(self.sm.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        data = glReadPixels(0, 0, 100, 100, GL_RGBA, GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape((100, 100, 4))

        # Center inside quad: UV=(0.50, 0.50) -> pixel (50, 50) -> full mask (255)
        self.assertEqual(img[50, 50, 0], 255)

        # To the left of the hand box: UV=(0.20, 0.54) -> pixel x=20, y=100-54=46
        # In the buggy implementation this was 255 (the phantom spike)!
        self.assertEqual(img[46, 20, 0], 0, "Spike glitch detected at UV (0.20, 0.54)")
        self.assertEqual(img[50, 25, 0], 0, "Spike glitch detected at UV (0.25, 0.50)")
        self.assertEqual(img[42, 15, 0], 0, "Spike glitch detected at UV (0.15, 0.58)")

        # Realistic coordinates from Image 1 (pinched right hand):
        # Left hand: thumb1=(0.39, 0.58), index1=(0.38, 0.40)
        # Right hand (pinched): thumb2=(0.59, 0.41), index2=(0.59, 0.39)
        prog.set_uniform_2f("u_thumb1", 0.39, 0.58)
        prog.set_uniform_2f("u_index1", 0.38, 0.40)
        prog.set_uniform_2f("u_thumb2", 0.59, 0.41)
        prog.set_uniform_2f("u_index2", 0.59, 0.39)

        glBindVertexArray(self.sm.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        data1 = glReadPixels(0, 0, 100, 100, GL_RGBA, GL_UNSIGNED_BYTE)
        img1 = np.frombuffer(data1, dtype=np.uint8).reshape((100, 100, 4))

        # Inside the wedge: UV=(0.48, 0.42) -> pixel x=48, y=100-42=58
        self.assertGreater(img1[58, 48, 0], 0)

        # Far left where phantom triangle appeared in Image 1: UV=(0.20, 0.55) -> pixel x=20, y=100-55=45
        self.assertEqual(img1[45, 20, 0], 0, "Image 1 phantom spike detected at UV (0.20, 0.55)")
        self.assertEqual(img1[42, 18, 0], 0, "Image 1 phantom apex detected at UV (0.18, 0.58)")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [tex])
        prog.delete()

    def test_recursive_includes_and_cycle_guard(self):
        """Verify recursive include resolution, max depth guard, and circular include protection (BUG-13)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            file_c = os.path.join(tmpdir, "c.glsl")
            with open(file_c, "w") as f:
                f.write("// C content\nfloat c_var = 1.0;\n")

            file_b = os.path.join(tmpdir, "b.glsl")
            with open(file_b, "w") as f:
                f.write('#include "c.glsl"\nfloat b_var = 2.0;\n')

            file_a = os.path.join(tmpdir, "a.glsl")
            with open(file_a, "w") as f:
                f.write('#include "b.glsl"\nfloat a_var = 3.0;\n')

            # Recursive resolution
            with open(file_a, "r") as f:
                src = f.read()
            resolved = self.sm._resolve_includes(src, tmpdir)
            self.assertIn("c_var = 1.0", resolved)
            self.assertIn("b_var = 2.0", resolved)
            self.assertIn("a_var = 3.0", resolved)

            # Circular include
            file_circ1 = os.path.join(tmpdir, "circ1.glsl")
            file_circ2 = os.path.join(tmpdir, "circ2.glsl")
            with open(file_circ1, "w") as f:
                f.write('#include "circ2.glsl"\n')
            with open(file_circ2, "w") as f:
                f.write('#include "circ1.glsl"\n')

            circ_resolved = self.sm._resolve_includes('#include "circ1.glsl"', tmpdir)
            self.assertIn("Circular include detected", circ_resolved)

            # Max depth guard
            deep_resolved = self.sm._resolve_includes('#include "b.glsl"', tmpdir, depth=10, max_depth=10)
            self.assertIn("Maximum include depth (10) exceeded", deep_resolved)

    def test_zero_intensity_passthrough_all_shaders(self):
        """Verify that zero intensity renders without GL error and triggers passthrough path (BUG-09, BUG-10)."""
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        for i in range(self.sm.shader_count):
            self.sm.select_shader(i)
            self.sm.render_quad(
                texture_id=tex,
                width=64,
                height=64,
                elapsed_time=1.0,
                hand_box=(0.2, 0.2, 0.8, 0.8),
                has_box=1.0,
                mask_mode=2,  # Fullscreen
                intensity=0.0,
                mouse_uv=(0.5, 0.5),
                thumb1=(0.2, 0.2), index1=(0.3, 0.2),
                thumb2=(0.8, 0.8), index2=(0.7, 0.8)
            )
            err = glGetError()
            self.assertEqual(err, GL_NO_ERROR, f"GL error {err} with zero intensity in {self.sm.active_shader_name}")

    def test_render_with_frozen_shapes_gpu(self):
        """Verify render_quad executes with multiple frozen shapes without GL errors."""
        from types import SimpleNamespace
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        frozen = [
            SimpleNamespace(is_quad=False, box=(0.1, 0.1, 0.3, 0.3), pts=[]),
            SimpleNamespace(
                is_quad=True,
                box=(0.6, 0.6, 0.9, 0.9),
                pts=[(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]
            ),
        ]

        for i in range(self.sm.shader_count):
            self.sm.select_shader(i)
            self.sm.render_quad(
                texture_id=tex,
                width=64,
                height=64,
                elapsed_time=1.0,
                hand_box=(0.4, 0.4, 0.6, 0.6),
                has_box=1.0,
                mask_mode=0,
                intensity=1.0,
                mouse_uv=(0.5, 0.5),
                thumb1=(0.4, 0.4), index1=(0.6, 0.4),
                thumb2=(0.6, 0.6), index2=(0.4, 0.6),
                frozen_shapes=frozen
            )
            err = glGetError()
            self.assertEqual(err, GL_NO_ERROR, f"GL error {err} with frozen shapes in {self.sm.active_shader_name}")

        glDeleteTextures(1, [tex])

    def test_frozen_shapes_mask_accumulation(self):
        """Verify that frozen shapes maintain their mask even when live hands are moved or removed."""
        import numpy as np
        from types import SimpleNamespace

        test_frag = """#version 330 core
        in vec2 v_uv;
        out vec4 FragColor;
        uniform vec4 u_hand_box;
        uniform float u_has_box;
        uniform int u_mask_mode;

        #include "common.glsl"

        void main() {
            float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);
            FragColor = vec4(mask, mask, mask, 1.0);
        }
        """
        from shader_manager import VERTEX_SHADER_SOURCE
        resolved = self.sm._resolve_includes(test_frag, self.sm.shaders_dir)
        prog = self.sm._compile_shader_from_source("test_frozen_mask", VERTEX_SHADER_SOURCE, resolved, "")
        self.assertIsNotNone(prog, "Failed to compile test_frozen_mask shader")

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 100, 100, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glViewport(0, 0, 100, 100)

        frozen = [
            # Frozen box at [0.60, 0.60, 0.90, 0.90]
            SimpleNamespace(is_quad=False, box=(0.60, 0.60, 0.90, 0.90), pts=[]),
            # Frozen quad at bottom-right
            SimpleNamespace(
                is_quad=True,
                box=(0.60, 0.10, 0.90, 0.40),
                pts=[(0.60, 0.10), (0.90, 0.10), (0.90, 0.40), (0.60, 0.40)]
            )
        ]

        # Case 1: Live hand is at [0.10, 0.10, 0.30, 0.30] AND 2 shapes are frozen
        prog.use()
        prog.set_uniform_2f("u_resolution", 100.0, 100.0)
        prog.set_uniform_1f("u_time", 0.0)
        prog.set_uniform_4f("u_hand_box", 0.10, 0.10, 0.30, 0.30)
        prog.set_uniform_1f("u_has_box", 1.0)
        prog.set_uniform_1i("u_mask_mode", 0)
        prog.set_uniform_2f("u_thumb1", 0.0, 0.0)
        prog.set_uniform_2f("u_index1", 0.0, 0.0)
        prog.set_uniform_2f("u_thumb2", 0.0, 0.0)
        prog.set_uniform_2f("u_index2", 0.0, 0.0)

        # Set frozen uniforms
        prog.set_uniform_1i("u_num_frozen", 2)
        prog.set_uniform_1i("u_frozen_is_quad[0]", 0)
        prog.set_uniform_4f("u_frozen_boxes[0]", 0.60, 0.60, 0.90, 0.90)
        prog.set_uniform_1i("u_frozen_is_quad[1]", 1)
        for j, pt in enumerate(frozen[1].pts):
            prog.set_uniform_2f(f"u_frozen_pts[{4 + j}]", pt[0], pt[1])

        glBindVertexArray(self.sm.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        data = glReadPixels(0, 0, 100, 100, GL_RGBA, GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape((100, 100, 4))

        # UV (0.20, 0.20) -> pixel x=20, y=100-20=80 -> Live box (active)
        self.assertEqual(img[80, 20, 0], 255, "Active live box should be masked")
        # UV (0.75, 0.75) -> pixel x=75, y=100-75=25 -> Frozen box
        self.assertEqual(img[25, 75, 0], 255, "Frozen box 0 should be masked")
        # UV (0.75, 0.25) -> pixel x=75, y=100-25=75 -> Frozen quad
        self.assertEqual(img[75, 75, 0], 255, "Frozen quad 1 should be masked")
        # UV (0.50, 0.50) -> pixel x=50, y=50 -> Outside all shapes
        self.assertEqual(img[50, 50, 0], 0, "Outside all shapes should have 0 mask")

        # Case 2: Hands dropped from camera (u_has_box = 0.0) -> Frozen shapes persist!
        prog.set_uniform_1f("u_has_box", 0.0)
        glBindVertexArray(self.sm.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        data2 = glReadPixels(0, 0, 100, 100, GL_RGBA, GL_UNSIGNED_BYTE)
        img2 = np.frombuffer(data2, dtype=np.uint8).reshape((100, 100, 4))

        # Live box should now have NO mask
        self.assertEqual(img2[80, 20, 0], 0, "Live box without hands must not have mask")
        # But frozen shapes remain masked!
        self.assertEqual(img2[25, 75, 0], 255, "Frozen box persists when hands dropped")
        self.assertEqual(img2[75, 75, 0], 255, "Frozen quad persists when hands dropped")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [tex])
        prog.delete()


if __name__ == "__main__":
    unittest.main()


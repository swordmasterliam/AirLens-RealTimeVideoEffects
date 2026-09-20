import math
import time
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *


HUD_VERT_SHADER = """#version 330 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;

void main() {
    // Normal OpenGL coordinates (-1, -1) to (1, 1)
    // Map to UV where (0, 0) is top-left, (1, 1) is bottom-right
    v_uv = vec2((a_pos.x + 1.0) * 0.5, (1.0 - a_pos.y) * 0.5);
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

HUD_FRAG_SHADER = """#version 330 core
out vec4 FragColor;
in vec2 v_uv;
uniform sampler2D u_hud_tex;

void main() {
    vec4 color = texture(u_hud_tex, v_uv);
    FragColor = color;
}
"""

RETICLE_VERT_SHADER = """#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_color;
out vec4 v_color;

void main() {
    // Map UV space [0, 1] to OpenGL clip space [-1, 1]
    gl_Position = vec4(a_pos.x * 2.0 - 1.0, 1.0 - a_pos.y * 2.0, 0.0, 1.0);
    v_color = a_color;
}
"""

RETICLE_FRAG_SHADER = """#version 330 core
in vec4 v_color;
out vec4 FragColor;

void main() {
    FragColor = v_color;
}
"""

POINT_VERT_SHADER = """#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_color;
out vec4 v_color;

void main() {
    gl_Position = vec4(a_pos.x * 2.0 - 1.0, 1.0 - a_pos.y * 2.0, 0.0, 1.0);
    v_color = a_color;
    gl_PointSize = 12.0;
}
"""

POINT_FRAG_SHADER = """#version 330 core
in vec4 v_color;
out vec4 FragColor;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    if (dist > 0.5) discard;
    float edge = 1.0 - smoothstep(0.32, 0.5, dist);
    FragColor = vec4(v_color.rgb, v_color.a * edge);
}
"""

DARK_GREEN_TITLE = (28, 150, 52, 255)       # Rich, crisp dark green for titles
DARK_GREEN_TEXT = (24, 140, 48, 255)        # Primary dark green for shader info, modes, buttons
DARK_GREEN_MUTED = (20, 128, 44, 240)       # Secondary dark green for metadata, key help
DARK_GREEN_ACCENT = (32, 165, 58, 255)      # Highlight dark green for active button & toasts
DARK_GREEN_LINE = (20, 120, 45, 120)        # Dark green accent dividing lines


def clamp(val, low, high):
    return max(low, min(val, high))


class HudRenderer:
    def __init__(self):
        self.visible = True
        self.show_box_visual = True

        self._texture_id = 0
        self._shader_prog = 0
        self._vao = 0
        self._vbo = 0

        self._reticle_prog = 0
        self._reticle_vao = 0
        self._reticle_vbo = 0
        self._point_prog = 0
        self._point_vao = 0
        self._point_vbo = 0

        self._width = 1280
        self._height = 720
        self._last_hud_update = 0.0
        self._update_interval = 0.08
        self._last_tracking_state = None
        self._gesture_toast = None

        self._mouse_x = -1.0
        self._mouse_y = -1.0
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._base_btn_rect = (200, 10, 435, 38)
        self._btn_rect = (200, 10, 435, 38)
        self.extra_gestures_enabled = True

        self._load_fonts()
        self._init_gl()
        self._init_reticle_gl()

    def _load_fonts(self):
        """Try loading system fonts, fall back to default."""
        font_paths = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/consola.ttf"
        ]
        self.font_large = None
        self.font_med = None
        self.font_small = None

        for path in font_paths:
            try:
                self.font_large = ImageFont.truetype(path, 22)
                self.font_med = ImageFont.truetype(path, 16)
                self.font_small = ImageFont.truetype(path, 13)
                break
            except Exception:
                continue

        if self.font_large is None:
            self.font_large = ImageFont.load_default()
            self.font_med = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def _compile_program(self, vert_src: str, frag_src: str) -> int:
        vert = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vert, vert_src)
        glCompileShader(vert)

        frag = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(frag, frag_src)
        glCompileShader(frag)

        prog = glCreateProgram()
        glAttachShader(prog, vert)
        glAttachShader(prog, frag)
        glLinkProgram(prog)

        glDeleteShader(vert)
        glDeleteShader(frag)
        return prog

    def _init_gl(self):
        self._texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        empty_data = np.zeros((self._height, self._width, 4), dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._width, self._height, 0, GL_RGBA, GL_UNSIGNED_BYTE, empty_data)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._shader_prog = self._compile_program(HUD_VERT_SHADER, HUD_FRAG_SHADER)

        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32)

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _init_reticle_gl(self):
        self._reticle_prog = self._compile_program(RETICLE_VERT_SHADER, RETICLE_FRAG_SHADER)
        self._point_prog = self._compile_program(POINT_VERT_SHADER, POINT_FRAG_SHADER)

        self._reticle_vao = glGenVertexArrays(1)
        self._reticle_vbo = glGenBuffers(1)

        glBindVertexArray(self._reticle_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._reticle_vbo)
        glBufferData(GL_ARRAY_BUFFER, 256 * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._point_vao = glGenVertexArrays(1)
        self._point_vbo = glGenBuffers(1)

        glBindVertexArray(self._point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._point_vbo)
        glBufferData(GL_ARRAY_BUFFER, 8 * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def toggle(self):
        self.visible = not self.visible

    def toggle_box_visual(self):
        self.show_box_visual = not self.show_box_visual

    def set_scale(self, scale_x: float = 1.0, scale_y: Optional[float] = None):
        if scale_y is None:
            scale_y = scale_x
        self._scale_x = scale_x
        self._scale_y = scale_y
        bx0, by0, bx1, by1 = self._base_btn_rect
        self._btn_rect = (bx0 * scale_x, by0 * scale_y, bx1 * scale_x, by1 * scale_y)

    def set_mouse_pos(self, x: float, y: float):
        self._mouse_x = x
        self._mouse_y = y

    def is_gesture_button_clicked(self, x: float, y: float, scale_x: Optional[float] = None, scale_y: Optional[float] = None) -> bool:
        if not self.visible:
            return False
        if scale_x is not None:
            sy = scale_y if scale_y is not None else scale_x
            bx0 = self._base_btn_rect[0] * scale_x
            by0 = self._base_btn_rect[1] * sy
            bx1 = self._base_btn_rect[2] * scale_x
            by1 = self._base_btn_rect[3] * sy
        else:
            bx0, by0, bx1, by1 = self._btn_rect
        return bx0 <= x <= bx1 and by0 <= y <= by1

    def notify_gesture(self, title: str, subtitle: str, color: Tuple[int, int, int] = (28, 150, 52), duration: float = 1.8):
        now = time.perf_counter()
        if self._gesture_toast and self._gesture_toast["title"] == title:
            self._gesture_toast["subtitle"] = subtitle
            self._gesture_toast["color"] = color
            self._gesture_toast["duration"] = max(self._gesture_toast["duration"], (now - self._gesture_toast["time"]) + duration)
        else:
            self._gesture_toast = {
                "title": title,
                "subtitle": subtitle,
                "color": color,
                "time": now,
                "duration": duration
            }

    def update_overlay(self, width: int, height: int,
                       shader_name: str, shader_idx: int, total_shaders: int,
                       mask_mode: int, intensity: float, mirrored: bool,
                       render_fps: float, render_ms: float,
                       camera_fps: float, camera_res: Tuple[int, int], camera_backend: str,
                       tracking_state, error_msg: Optional[str] = None,
                       force: bool = False,
                       extra_gestures_enabled: bool = True,
                       boost_mode: bool = False,
                       frozen_shapes: Optional[list] = None):
        """Re-render HUD text panels to texture if interval elapsed or forced."""
        self._last_tracking_state = tracking_state
        self._frozen_shapes = frozen_shapes or []
        self.extra_gestures_enabled = extra_gestures_enabled
        self.boost_mode = boost_mode or (tracking_state and getattr(tracking_state, "boost_active", False))

        now = time.perf_counter()
        if not force and (now - self._last_hud_update) < self._update_interval:
            return

        self._last_hud_update = now
        self._width = max(width, 100)
        self._height = max(height, 100)

        img = Image.new("RGBA", (self._width, self._height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if self.visible:
            bar_height = 82
            draw.rectangle([0, 0, self._width, bar_height], fill=(12, 16, 24, 205))
            draw.line([(0, bar_height), (self._width, bar_height)], fill=DARK_GREEN_LINE, width=1)

            # Application Title
            draw.text((20, 12), "PLuhh on cam", fill=DARK_GREEN_TITLE, font=self.font_large)
            if self.boost_mode:
                draw.text((160, 15), "[BOOST ACTIVE]", fill=(40, 240, 90, 255), font=self.font_small)

            # Interactive Gesture Toggle Button
            bx0, by0, bx1, by1 = self._btn_rect
            is_hovered = (bx0 <= self._mouse_x <= bx1 and by0 <= self._mouse_y <= by1)
            if extra_gestures_enabled:
                btn_bg = (18, 42, 22, 230) if is_hovered else (12, 30, 16, 200)
                btn_border = DARK_GREEN_ACCENT if is_hovered else DARK_GREEN_TEXT
                btn_border_w = 2 if is_hovered else 1
                btn_text = "[G] Disable Extra Gestures"
                btn_col = DARK_GREEN_ACCENT if is_hovered else DARK_GREEN_TEXT
            else:
                btn_bg = (24, 56, 30, 240) if is_hovered else (18, 44, 22, 220)
                btn_border = (38, 175, 62, 255) if is_hovered else DARK_GREEN_TITLE
                btn_border_w = 2
                btn_text = "[G] Gestures: Box Only"
                btn_col = DARK_GREEN_ACCENT

            draw.rectangle([bx0, by0, bx1, by1], fill=btn_bg, outline=btn_border, width=btn_border_w)
            draw.text((bx0 + 10, by0 + 6), btn_text, fill=btn_col, font=self.font_small)

            shader_text = f"[{shader_idx + 1}/{total_shaders}] {shader_name}"
            draw.text((20, 44), shader_text, fill=DARK_GREEN_TEXT, font=self.font_med)

            mode_names = [
                "Mask: Hand Box [Key M]",
                "Mask: Inverted Box [Key M]",
                "Mask: Fullscreen [Key M]"
            ]
            draw.text((360, 44), mode_names[mask_mode], fill=DARK_GREEN_TEXT, font=self.font_med)

            frozen_count = len(self._frozen_shapes)
            if frozen_count > 0:
                frozen_label = f"Frozen: {frozen_count}/8 [C: Clear | Z: Undo]"
                draw.text((600, 44), frozen_label, fill=(40, 230, 180, 255), font=self.font_med)
                intensity_text = f"Intensity: {intensity:.2f}"
                draw.text((840, 44), intensity_text, fill=DARK_GREEN_TEXT, font=self.font_med)
            else:
                intensity_text = f"Intensity: {intensity:.2f} [Up/Down]"
                draw.text((600, 44), intensity_text, fill=DARK_GREEN_TEXT, font=self.font_med)

            fps_text = f"Render: {render_fps:5.1f} FPS ({render_ms:4.1f} ms) | GPU Native"
            draw.text((self._width - 410, 12), fps_text, fill=DARK_GREEN_TEXT, font=self.font_med)

            if "Mock" in camera_backend:
                backend_label = "Synthetic Mock Pattern"
            elif "DSHOW" in camera_backend or "DirectShow" in camera_backend:
                backend_label = "DirectShow buf=1"
            elif "MSMF" in camera_backend:
                backend_label = "Media Foundation buf=1"
            else:
                backend_label = f"{camera_backend} buf=1"

            cam_text = f"Camera: {camera_res[0]}x{camera_res[1]} @ {camera_fps:4.1f} FPS ({backend_label})"
            draw.text((self._width - 410, 34), cam_text, fill=DARK_GREEN_MUTED, font=self.font_small)

            hands_count = tracking_state.hands_count if tracking_state else 0
            infer_ms = tracking_state.inference_time_ms if tracking_state else 0.0
            track_fps = tracking_state.tracking_fps if tracking_state else 0.0
            raw_g = getattr(tracking_state, "active_gesture", "none")
            gesture_label = raw_g.replace("_", " ").upper() if (raw_g and raw_g != "none") else "NONE"
            hands_status = f"Hands: {hands_count} ({infer_ms:3.1f} ms) | Gesture: {gesture_label}"
            draw.text((self._width - 410, 52), hands_status, fill=DARK_GREEN_TEXT, font=self.font_small)

            bar_h = 44
            key_bar_y = self._height - bar_h
            draw.rectangle([0, key_bar_y, self._width, self._height], fill=(12, 16, 24, 210))
            draw.line([(0, key_bar_y), (self._width, key_bar_y)], fill=DARK_GREEN_LINE, width=1)
            key_help = "KEYS: SPACE/TAB: Next | 1-0: Select | M: Mask | W/Wink: Freeze | C: Clear | Z: Undo | B: Reticle | V: Boost | ESC: Exit"
            if extra_gestures_enabled:
                gesture_help = "GESTURES: Snap: Next | Peace: Mask | Fist: Reticle | Horns: Boost | Wink: Freeze Shape"
            else:
                gesture_help = "GESTURES: Extra gestures disabled (Box only active) | Wink/W: Freeze Shape | C: Clear"
            draw.text((20, key_bar_y + 4), key_help, fill=DARK_GREEN_MUTED, font=self.font_small)
            draw.text((20, key_bar_y + 22), gesture_help, fill=DARK_GREEN_TEXT, font=self.font_small)

        if self._gesture_toast:
            toast_age = now - self._gesture_toast["time"]
            toast_dur = self._gesture_toast["duration"]
            if toast_age < toast_dur:
                if toast_age < 0.12:
                    alpha = toast_age / 0.12
                elif toast_age < toast_dur * 0.65:
                    alpha = 1.0
                else:
                    alpha = max(0.0, (toast_dur - toast_age) / (toast_dur * 0.35))

                alpha_int = int(255 * alpha)
                if alpha_int > 5:
                    bw, bh = 460, 54
                    bx = (self._width - bw) // 2
                    by = 92

                    draw.rectangle([bx, by, bx + bw, by + bh],
                                   fill=(14, 24, 16, int(225 * alpha)),
                                   outline=(DARK_GREEN_ACCENT[0], DARK_GREEN_ACCENT[1], DARK_GREEN_ACCENT[2], int(240 * alpha)),
                                   width=2)

                    draw.rectangle([bx, by, bx + 6, by + bh],
                                   fill=(DARK_GREEN_ACCENT[0], DARK_GREEN_ACCENT[1], DARK_GREEN_ACCENT[2], int(255 * alpha)))

                    draw.text((bx + 18, by + 6), self._gesture_toast["title"], fill=(DARK_GREEN_ACCENT[0], DARK_GREEN_ACCENT[1], DARK_GREEN_ACCENT[2], alpha_int), font=self.font_med)
                    draw.text((bx + 18, by + 28), self._gesture_toast["subtitle"], fill=(DARK_GREEN_TEXT[0], DARK_GREEN_TEXT[1], DARK_GREEN_TEXT[2], alpha_int), font=self.font_small)
            else:
                self._gesture_toast = None

        if error_msg:
            err_box_y = 154 if self._gesture_toast else 90
            draw.rectangle([10, err_box_y, self._width - 10, err_box_y + 60], fill=(20, 36, 18, 220), outline=(DARK_GREEN_ACCENT[0], DARK_GREEN_ACCENT[1], DARK_GREEN_ACCENT[2], 255), width=2)
            draw.text((20, err_box_y + 8), "SHADER COMPILATION ERROR (Previous shader kept active):", fill=DARK_GREEN_ACCENT, font=self.font_med)
            draw.text((20, err_box_y + 30), error_msg[:140], fill=DARK_GREEN_TEXT, font=self.font_small)

        if self._texture_id == 0:
            return

        img_bytes = img.tobytes()
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        if not hasattr(self, "_tex_allocated_w") or self._tex_allocated_w != self._width or self._tex_allocated_h != self._height:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._width, self._height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_bytes)
            self._tex_allocated_w = self._width
            self._tex_allocated_h = self._height
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self._width, self._height, GL_RGBA, GL_UNSIGNED_BYTE, img_bytes)
        glBindTexture(GL_TEXTURE_2D, 0)

    def render(self, tracking_state=None, frozen_shapes=None):
        if not self._reticle_prog and not self._texture_id:
            return

        state = tracking_state or self._last_tracking_state
        shapes_to_draw = frozen_shapes if frozen_shapes is not None else getattr(self, "_frozen_shapes", [])

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)

        has_live = state and getattr(state, "has_box", 0.0) > 0.02
        has_frozen = bool(shapes_to_draw and len(shapes_to_draw) > 0)

        if (self.show_box_visual and (has_live or has_frozen)
                and self._reticle_prog and self._reticle_vao and self._reticle_vbo):
            self._render_gpu_reticle(state, shapes_to_draw)

        if self.visible and self._texture_id != 0 and self._shader_prog != 0 and self._vao != 0:
            glUseProgram(self._shader_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._texture_id)
            loc = glGetUniformLocation(self._shader_prog, "u_hud_tex")
            if loc != -1:
                glUniform1i(loc, 0)

            glBindVertexArray(self._vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)
            glUseProgram(0)
            glBindTexture(GL_TEXTURE_2D, 0)

        glDisable(GL_BLEND)

    @staticmethod
    def _is_valid_point(pt) -> bool:
        if pt is None or len(pt) < 2:
            return False
        return abs(pt[0]) > 0.001 or abs(pt[1]) > 0.001

    def _compute_reticle_lines(self, state, alpha: float, frozen_shapes=None):
        lines = []

        # 1. Compute lines for frozen shapes (in distinct warm gold color)
        if frozen_shapes:
            col_frozen = (1.0, 0.82, 0.22, 0.85)
            for f_shape in frozen_shapes[:8]:
                if getattr(f_shape, "is_quad", False):
                    pts = getattr(f_shape, "pts", [])
                    if len(pts) >= 4:
                        p0, p1, p2, p3 = pts[:4]
                        lines.extend([
                            (p0[0], p0[1], *col_frozen), (p1[0], p1[1], *col_frozen),
                            (p1[0], p1[1], *col_frozen), (p2[0], p2[1], *col_frozen),
                            (p2[0], p2[1], *col_frozen), (p3[0], p3[1], *col_frozen),
                            (p3[0], p3[1], *col_frozen), (p0[0], p0[1], *col_frozen),
                        ])
                else:
                    fbox = getattr(f_shape, "box", None)
                    if fbox and len(fbox) >= 4:
                        x0, y0, x1, y1 = fbox
                        lines.extend([
                            (x0, y0, *col_frozen), (x1, y0, *col_frozen),
                            (x1, y0, *col_frozen), (x1, y1, *col_frozen),
                            (x1, y1, *col_frozen), (x0, y1, *col_frozen),
                            (x0, y1, *col_frozen), (x0, y0, *col_frozen),
                        ])

        # 2. Compute lines for active live shape (cyan)
        if state and alpha > 0.02:
            t1 = getattr(state, "thumb1", None)
            i1 = getattr(state, "index1", None)
            t2 = getattr(state, "thumb2", None)
            i2 = getattr(state, "index2", None)

            has_t1 = self._is_valid_point(t1)
            has_i1 = self._is_valid_point(i1)
            has_t2 = self._is_valid_point(t2)
            has_i2 = self._is_valid_point(i2)

            hands_count = getattr(state, "hands_count", 0)
            four_fingers = has_t1 and has_i1 and has_t2 and has_i2 and (
                hands_count >= 2 or (hands_count == 0 and has_t2)
            )

            col = (0.0, 0.9, 1.0, alpha * 0.85)

            if four_fingers:
                pts = [t1, i1, t2, i2]
                cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25
                cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25
                sorted_pts = sorted(
                    pts,
                    key=lambda p: (
                        math.atan2(p[1] - cy, p[0] - cx),
                        (p[0] - cx) ** 2 + (p[1] - cy) ** 2
                    )
                )
                p0, p1, p2, p3 = sorted_pts
                lines.extend([
                    (p0[0], p0[1], *col), (p1[0], p1[1], *col),
                    (p1[0], p1[1], *col), (p2[0], p2[1], *col),
                    (p2[0], p2[1], *col), (p3[0], p3[1], *col),
                    (p3[0], p3[1], *col), (p0[0], p0[1], *col),
                ])
            else:
                box = getattr(state, "box", (0.25, 0.25, 0.75, 0.75))
                if not box or len(box) < 4:
                    box = (0.25, 0.25, 0.75, 0.75)
                x0, y0, x1, y1 = box
                lines.extend([
                    (x0, y0, *col), (x1, y0, *col),
                    (x1, y0, *col), (x1, y1, *col),
                    (x1, y1, *col), (x0, y1, *col),
                    (x0, y1, *col), (x0, y0, *col),
                ])

        return lines

    def _compute_reticle_points(self, state, alpha: float, frozen_shapes=None):
        points = []

        # Frozen points in warm gold
        if frozen_shapes:
            col_fpt = (1.0, 0.82, 0.22, 0.85)
            for f_shape in frozen_shapes[:8]:
                if getattr(f_shape, "is_quad", False):
                    for pt in getattr(f_shape, "pts", [])[:4]:
                        points.append((pt[0], pt[1], *col_fpt))

        # Live shape points
        if state and alpha > 0.02:
            col_t1 = (1.0, 0.85, 0.2, alpha * 0.9)
            col_t2 = (0.0, 0.9, 1.0, alpha * 0.9)

            t1 = getattr(state, "thumb1", None)
            i1 = getattr(state, "index1", None)
            t2 = getattr(state, "thumb2", None)
            i2 = getattr(state, "index2", None)

            if self._is_valid_point(t1):
                points.append((t1[0], t1[1], *col_t1))
            if self._is_valid_point(i1):
                points.append((i1[0], i1[1], *col_t1))
            if self._is_valid_point(t2):
                points.append((t2[0], t2[1], *col_t2))
            if self._is_valid_point(i2):
                points.append((i2[0], i2[1], *col_t2))

        return points

    def _render_gpu_reticle(self, state, frozen_shapes=None):
        if not self._reticle_prog or not self._reticle_vao or not self._reticle_vbo:
            return

        alpha = float(clamp(getattr(state, "has_box", 0.0) if state else 0.0, 0.0, 1.0))
        lines = self._compute_reticle_lines(state, alpha, frozen_shapes)
        if not lines:
            return

        line_data = np.array(lines, dtype=np.float32)

        glUseProgram(self._reticle_prog)
        glBindVertexArray(self._reticle_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._reticle_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, line_data.nbytes, line_data)
        glDrawArrays(GL_LINES, 0, len(lines))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        points = self._compute_reticle_points(state, alpha, frozen_shapes)

        if points and self._point_prog and self._point_vao and self._point_vbo:
            pt_data = np.array(points, dtype=np.float32)
            glEnable(GL_PROGRAM_POINT_SIZE)
            glUseProgram(self._point_prog)
            glBindVertexArray(self._point_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._point_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, pt_data.nbytes, pt_data)
            glDrawArrays(GL_POINTS, 0, len(points))
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            glDisable(GL_PROGRAM_POINT_SIZE)

        glUseProgram(0)

    def cleanup(self):
        if self._texture_id:
            glDeleteTextures(1, [self._texture_id])
            self._texture_id = 0
        if self._shader_prog:
            glDeleteProgram(self._shader_prog)
            self._shader_prog = 0
        if self._vbo:
            glDeleteBuffers(1, [self._vbo])
            self._vbo = 0
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
            self._vao = 0
        if self._reticle_prog:
            glDeleteProgram(self._reticle_prog)
            self._reticle_prog = 0
        if self._reticle_vbo:
            glDeleteBuffers(1, [self._reticle_vbo])
            self._reticle_vbo = 0
        if self._reticle_vao:
            glDeleteVertexArrays(1, [self._reticle_vao])
            self._reticle_vao = 0
        if self._point_prog:
            glDeleteProgram(self._point_prog)
            self._point_prog = 0
        if self._point_vbo:
            glDeleteBuffers(1, [self._point_vbo])
            self._point_vbo = 0
        if self._point_vao:
            glDeleteVertexArrays(1, [self._point_vao])
            self._point_vao = 0

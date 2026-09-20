import sys
import os
import time
import argparse
import subprocess
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import glfw
from OpenGL.GL import *

from camera import CameraCapture
from tracker import HandTracker, TrackingData
from shader_manager import ShaderManager
from hud import HudRenderer
from gesture import GestureType


@dataclass
class FrozenShape:
    is_quad: bool
    box: Tuple[float, float, float, float]
    pts: List[Tuple[float, float]]


class VideoEffectsApp:
    def __init__(self, camera_index=0, width=1280, height=720, mock=False, vsync=True, max_frames=0, boost=False):
        self.preferred_width = width
        self.preferred_height = height
        self.camera_index = camera_index
        self.mock = mock
        self.vsync = vsync
        self.max_frames = max_frames
        self.boost_mode = boost

        self.mask_mode = 0 #0 =box 1 = outside box, like inverted, 2 = fullscreen
        self.intensity = 1.0
        self.mirrored = True
        self.mouse_uv = (0.5, 0.5)
        self.window_width = width
        self.window_height = height

        self.is_fullscreen = False

        self._windowed_pos = (100, 100)
        self._windowed_size = (width, height)

        self.frozen_shapes: List[FrozenShape] = []

        self.render_fps = 0.0
        self.render_ms = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
        self._total_frames_rendered = 0

        self.window = None
        self.camera = None
        self.tracker = None
        self.shader_manager = None
        self.hud = None

        self.extra_gestures_enabled = True

        self.cam_texture = 0
        self._cam_tex_allocated = False
        self._cam_tex_w = 0
        self._cam_tex_h = 0
        self._last_uploaded_frame_id = -1

    def toggle_boost_mode(self):
        self.boost_mode = not self.boost_mode
        if self.tracker:
            self.tracker.set_boost(self.boost_mode)
        if self.hud:
            if self.boost_mode:
                self.hud.notify_gesture(
                    "BOOST ACTIVATED",
                    "High-performance tracking und gestures enabled",
                    color=(32, 220, 90),
                    duration=2.0
                )
            else:
                self.hud.notify_gesture(
                    "BOOST DISABLED",
                    "Standard tracking mode",
                    color=(28, 150, 52),
                    duration=2.0
                )
        self._trigger_hud_update()

    def toggle_extra_gestures(self):
        self.extra_gestures_enabled = not self.extra_gestures_enabled
        if self.tracker:
            self.tracker.set_extra_gestures_enabled(self.extra_gestures_enabled)
        if self.hud:
            if not self.extra_gestures_enabled:
                self.hud.notify_gesture(
                    "GESTURES DISABLED",
                    "Only main hand box gesture will be active",
                    color=(28, 150, 52),
                    duration=2.0
                )
            else:
                self.hud.notify_gesture(
                    "GESTURES ENABLED",
                    "All hand gestures will be active",
                    color=(28, 150, 52),
                    duration=2.0
                )
        self._trigger_hud_update()

    def freeze_current_shape(self):
        tracking = self.tracker.get_state() if self.tracker else None
        if not tracking or tracking.has_box < 0.1:
            if self.hud:
                self.hud.notify_gesture(
                    "CANNOt freeze",
                    "No active hand frame detected",
                    color=(200, 160, 40),
                    duration=1.5
                )
            return

        if len(self.frozen_shapes) >= 8:
            if self.hud:
                self.hud.notify_gesture(
                    "MAX SHAPES BEEN REACHED",
                    "Limit is 8 frozen shapes (Press C to clear)",
                    color=(220, 80, 50),
                    duration=2.0
                )
            return

        t1 = tracking.thumb1
        i1 = tracking.index1
        t2 = tracking.thumb2
        i2 = tracking.index2


        def is_valid_pt(pt):
            return pt is not None and len(pt) >= 2 and (abs(pt[0]) > 0.001 or abs(pt[1]) > 0.001)

        four_fingers = (
            is_valid_pt(t1) and is_valid_pt(i1) and is_valid_pt(t2) and is_valid_pt(i2)
            and (tracking.hands_count >= 2 or (tracking.hands_count == 0 and is_valid_pt(t2)))
        )

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
            frozen = FrozenShape(
                is_quad=True,
                box=tracking.box,
                pts=sorted_pts
            )
        else:
            bx = tracking.box
            frozen = FrozenShape(
                is_quad=False,
                box=bx,
                pts=[(bx[0], bx[1]), (bx[2], bx[1]), (bx[2], bx[3]), (bx[0], bx[3])]
            )

        self.frozen_shapes.append(frozen)
        count = len(self.frozen_shapes)
        if self.hud:
            self.hud.notify_gesture(
                "SHAPE FROZEN (WINK)",
                f"Frozen shapes: {count}/8 [W: Freeze | C: Clear | Z: Undo]",
                color=(255, 210, 60),
                duration=2.5
            )
        self._trigger_hud_update()

    def clear_frozen_shapes(self):
        if not self.frozen_shapes:
            return
        self.frozen_shapes.clear()
        if self.hud:
            self.hud.notify_gesture(
                "SHAPES CLEARED",
                "All frozen shapes now cleared",
                color=(220, 180, 50),
                duration=1.5
            )
        self._trigger_hud_update()

    def undo_frozen_shape(self):
        if not self.frozen_shapes:
            return
        self.frozen_shapes.pop()
        count = len(self.frozen_shapes)
        if self.hud:
            self.hud.notify_gesture(
                "SHAPE UNDONE",
                f"Remaining frozen shapes: {count}/8",
                color=(220, 180, 50),
                duration=1.5
            )
        self._trigger_hud_update()

    def init_glfw_and_gl(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, GL_TRUE)

        if self.max_frames > 0 and "--headless" in sys.argv:
            glfw.window_hint(glfw.VISIBLE, GL_FALSE)

        self.window = glfw.create_window(
            self.preferred_width, self.preferred_height,
            "PLuhh", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")


        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.vsync else 0)


        renderer = glGetString(GL_RENDERER).decode("utf-8")
        version = glGetString(GL_VERSION).decode("utf-8")
        glsl_ver = glGetString(GL_SHADING_LANGUAGE_VERSION).decode("utf-8")
        print(f"[Engine] GPU Renderer: {renderer}")
        print(f"[Engine] OpenGL Version: {version}")
        print(f"[Engine] GLSL Version: {glsl_ver}")


        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)


        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.window_width = max(fb_w, 1)
        self.window_height = max(fb_h, 1)
        glViewport(0, 0, self.window_width, self.window_height)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glClearColor(0.05, 0.05, 0.08, 1.0)

        self.cam_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.cam_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _get_dpi_scale(self):
        if not self.window:
            return 1.0, 1.0
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        sx = (fb_w / win_w) if win_w > 0 else 1.0
        sy = (fb_h / win_h) if win_h > 0 else 1.0
        return sx, sy

    def _on_resize(self, window, w, h):
        self.window_width = max(w, 1)
        self.window_height = max(h, 1)
        glViewport(0, 0, self.window_width, self.window_height)
        if self.hud:
            sx, sy = self._get_dpi_scale()
            self.hud.set_scale(sx, sy)

    def _on_mouse_move(self, window, xpos, ypos):
        win_w, win_h = glfw.get_window_size(window)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        scale_x = (fb_w / win_w) if win_w > 0 else 1.0
        scale_y = (fb_h / win_h) if win_h > 0 else 1.0
        pixel_x = xpos * scale_x
        pixel_y = ypos * scale_y

        if win_w > 0 and win_h > 0:
            self.mouse_uv = (xpos / win_w, ypos / win_h)
        if self.hud:
            self.hud.set_mouse_pos(pixel_x, pixel_y)

    def _on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            win_w, win_h = glfw.get_window_size(window)
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            scale_x = (fb_w / win_w) if win_w > 0 else 1.0
            scale_y = (fb_h / win_h) if win_h > 0 else 1.0
            pixel_x = xpos * scale_x
            pixel_y = ypos * scale_y
            if self.hud and self.hud.is_gesture_button_clicked(pixel_x, pixel_y):
                self.toggle_extra_gestures()

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS and action != glfw.REPEAT:
            return

        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)
        elif key in (glfw.KEY_SPACE, glfw.KEY_TAB):
            if mods & glfw.MOD_SHIFT:
                self.shader_manager.prev_shader()
            else:
                self.shader_manager.next_shader()
            self._trigger_hud_update()
        elif key == glfw.KEY_BACKSPACE:
            self.shader_manager.prev_shader()
            self._trigger_hud_update()
        elif glfw.KEY_1 <= key <= glfw.KEY_9:
            idx = key - glfw.KEY_1
            self.shader_manager.select_shader(idx)
            self._trigger_hud_update()
        elif key == glfw.KEY_0:
            self.shader_manager.select_shader(9)
            self._trigger_hud_update()
        elif key == glfw.KEY_M and action == glfw.PRESS:
            self.mask_mode = (self.mask_mode + 1) % 3
            self._trigger_hud_update()
        elif key == glfw.KEY_B and action == glfw.PRESS:
            self.hud.toggle_box_visual()
            self._trigger_hud_update()
        elif key == glfw.KEY_V and action == glfw.PRESS:
            self.toggle_boost_mode()
        elif key == glfw.KEY_G and action == glfw.PRESS:
            self.toggle_extra_gestures()
        elif key == glfw.KEY_H and action == glfw.PRESS:
            self.hud.toggle()
            self._trigger_hud_update()
        elif key == glfw.KEY_X and action == glfw.PRESS:
            self.mirrored = not self.mirrored
            if self.camera:
                self.camera.set_mirrored(self.mirrored)
            self._trigger_hud_update()
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.freeze_current_shape()
        elif key == glfw.KEY_C and action == glfw.PRESS:
            self.clear_frozen_shapes()
        elif key == glfw.KEY_Z and action == glfw.PRESS:
            self.undo_frozen_shape()
        elif key == glfw.KEY_UP:
            self.intensity = min(3.0, self.intensity + 0.05)
            self._trigger_hud_update()
        elif key == glfw.KEY_DOWN:
            self.intensity = max(0.0, self.intensity - 0.05)
            self._trigger_hud_update()
        elif key == glfw.KEY_R and action == glfw.PRESS:
            print("[Engine] Manual hot-reload requested.")
            self.shader_manager.reload_all_shaders()
            self._trigger_hud_update()
        elif key == glfw.KEY_O and action == glfw.PRESS:
            shaders_path = self.shader_manager.shaders_dir
            print(f"[Engine] Opening shaders directory: {shaders_path}")
            if sys.platform == "win32":
                subprocess.Popen(["explorer", shaders_path])
        elif key == glfw.KEY_F and action == glfw.PRESS:
            self._toggle_fullscreen()

    def _get_current_monitor(self):
        if not self.window:
            return glfw.get_primary_monitor()
        monitors = glfw.get_monitors()
        if not monitors:
            return glfw.get_primary_monitor()
        if len(monitors) == 1:
            return monitors[0]

        wx, wy = glfw.get_window_pos(self.window)
        ww, wh = glfw.get_window_size(self.window)

        best_monitor = monitors[0]
        best_overlap = -1

        for monitor in monitors:
            mx, my = glfw.get_monitor_pos(monitor)
            mode = glfw.get_video_mode(monitor)
            if not mode:
                continue
            mw, mh = mode.size.width, mode.size.height

            overlap_x = max(0, min(wx + ww, mx + mw) - max(wx, mx))
            overlap_y = max(0, min(wy + wh, my + mh) - max(wy, my))
            overlap_area = overlap_x * overlap_y

            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_monitor = monitor

        return best_monitor

    def _toggle_fullscreen(self):
        if not self.window:
            return
        if not self.is_fullscreen:
            self._windowed_pos = glfw.get_window_pos(self.window)
            self._windowed_size = glfw.get_window_size(self.window)
            monitor = self._get_current_monitor()
            if monitor:
                mode = glfw.get_video_mode(monitor)
                if mode:
                    glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
                    self.is_fullscreen = True
        else:
            wx, wy = self._windowed_pos
            ww, wh = self._windowed_size
            glfw.set_window_monitor(self.window, None, wx, wy, ww, wh, 0)
            self.is_fullscreen = False

    def _trigger_hud_update(self):
        if self.hud:
            cam_res = self.camera.resolution if self.camera else (640, 480)
            tracking = self.tracker.get_state() if self.tracker else TrackingData()
            self.hud.update_overlay(
                self.window_width, self.window_height,
                self.shader_manager.active_shader_name,
                self.shader_manager.active_index,
                self.shader_manager.shader_count,
                self.mask_mode, self.intensity, self.mirrored,
                self.render_fps, self.render_ms,
                self.camera.fps if self.camera else 0.0,
                cam_res,
                self.camera.backend_name if self.camera else "None",
                tracking,
                self.shader_manager.last_error,
                force=True,
                extra_gestures_enabled=self.extra_gestures_enabled,
                boost_mode=self.boost_mode,
                frozen_shapes=self.frozen_shapes
            )

    def _handle_gesture_event(self, event):
        if not self.extra_gestures_enabled:
            return
        gtype = event.type
        if gtype == GestureType.SNAP:
            self.shader_manager.next_shader()
            self.hud.notify_gesture(
                "SNAP DETECTED",
                f"Next Shader: {self.shader_manager.active_shader_name}",
                color=(28, 150, 52)
            )
            self._trigger_hud_update()
        elif gtype == GestureType.PEACE:
            self.mask_mode = (self.mask_mode + 1) % 3
            mode_names = ["Box Only", "Inverted Box", "Fullscreen"]
            self.hud.notify_gesture(
                "PEACE SIGN",
                f"Mask Mode: {mode_names[self.mask_mode]}",
                color=(28, 150, 52)
            )
            self._trigger_hud_update()
        elif gtype == GestureType.FIST:
            self.hud.toggle_box_visual()
            status = "Visible" if self.hud.show_box_visual else "Hidden"
            self.hud.notify_gesture(
                "CLOSED FIST",
                f"Box Reticle: {status}",
                color=(28, 150, 52)
            )
            self._trigger_hud_update()
        elif gtype == GestureType.OPEN_PALM:
            self.intensity = 1.0
            self.hud.notify_gesture(
                "OPEN PALM",
                "Intensity Reset to 1.00",
                color=(28, 150, 52)
            )
            self._trigger_hud_update()
        elif gtype == GestureType.BOOST:
            self.toggle_boost_mode()

    def _handle_continuous_gesture(self, gesture, dt: float):
        if not self.extra_gestures_enabled:
            return
        rate = 1.0 if self.boost_mode else 0.5
        if gesture == GestureType.THUMBS_UP:
            self.intensity = min(3.0, self.intensity + rate * dt)
            self.hud.notify_gesture(
                "THUMBS UP",
                f"Intensity: {self.intensity:.2f} (Increasing)",
                color=(28, 150, 52),
                duration=0.6
            )
        elif gesture == GestureType.THUMBS_DOWN:
            self.intensity = max(0.0, self.intensity - rate * dt)
            self.hud.notify_gesture(
                "THUMBS DOWN",
                f"Intensity: {self.intensity:.2f} (Decreasing)",
                color=(28, 150, 52),
                duration=0.6
            )

    def _upload_camera_frame_to_gpu(self):
        if not self.camera or not self.cam_texture:
            return
        frame, frame_id, _ = self.camera.get_latest_frame()
        if frame is None or frame_id == self._last_uploaded_frame_id:
            return

        self._last_uploaded_frame_id = frame_id
        fh, fw = frame.shape[:2]

        glBindTexture(GL_TEXTURE_2D, self.cam_texture)

        if not self._cam_tex_allocated or self._cam_tex_w != fw or self._cam_tex_h != fh:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fw, fh, 0, GL_BGR, GL_UNSIGNED_BYTE, frame)
            self._cam_tex_w = fw
            self._cam_tex_h = fh
            self._cam_tex_allocated = True
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fw, fh, GL_BGR, GL_UNSIGNED_BYTE, frame)

        glBindTexture(GL_TEXTURE_2D, 0)

    def run(self):
        try:
            self.camera = CameraCapture(
                camera_index=self.camera_index,
                preferred_width=self.preferred_width,
                preferred_height=self.preferred_height,
                mirrored=self.mirrored,
                mock=self.mock
            )
            self.camera.start()

            self.tracker = HandTracker(self.camera)
            if self.boost_mode:
                self.tracker.set_boost(True)
            self.tracker.start()

            self.init_glfw_and_gl()

            shaders_dir = os.path.join(os.path.dirname(__file__), "shaders")
            self.shader_manager = ShaderManager(shaders_dir)
            self.hud = HudRenderer()
            sx, sy = self._get_dpi_scale()
            self.hud.set_scale(sx, sy)

            start_time = time.perf_counter()
            last_reload_check = time.perf_counter()
            prev_frame_time = time.perf_counter()

            print("[Engine] Rendering started. Press H for help, Space for next shader, Esc to quit.")

            while self.window and not glfw.window_should_close(self.window):
                t_frame_start = time.perf_counter()
                dt = max(0.0001, t_frame_start - prev_frame_time)
                prev_frame_time = t_frame_start
                elapsed_time = t_frame_start - start_time

                if (t_frame_start - last_reload_check) > 0.5:
                    last_reload_check = t_frame_start
                    if self.shader_manager and self.shader_manager.check_hot_reload():
                        self._trigger_hud_update()

                if self.tracker:
                    if self.tracker.poll_wink_event():
                        self.freeze_current_shape()

                    gesture_events = self.tracker.poll_gesture_events()
                    for g_event in gesture_events:
                        self._handle_gesture_event(g_event)

                    cont_gesture = self.tracker.get_continuous_gesture()
                    if cont_gesture:
                        self._handle_continuous_gesture(cont_gesture, dt)

                if self.camera:
                    self._upload_camera_frame_to_gpu()

                tracking_state = self.tracker.get_state() if self.tracker else TrackingData()

                glClear(GL_COLOR_BUFFER_BIT)

                if self.shader_manager:
                    self.shader_manager.render_quad(
                        texture_id=self.cam_texture,
                        width=self.window_width,
                        height=self.window_height,
                        elapsed_time=elapsed_time,
                        hand_box=tracking_state.box,
                        has_box=tracking_state.has_box,
                        mask_mode=self.mask_mode,
                        intensity=self.intensity,
                        mouse_uv=self.mouse_uv,
                        thumb1=tracking_state.thumb1,
                        index1=tracking_state.index1,
                        thumb2=tracking_state.thumb2,
                        index2=tracking_state.index2,
                        frozen_shapes=self.frozen_shapes
                    )

                if self.hud:
                    cam_res = self.camera.resolution if self.camera else (640, 480)
                    cam_fps = self.camera.fps if self.camera else 0.0
                    cam_backend = self.camera.backend_name if self.camera else "None"
                    active_name = self.shader_manager.active_shader_name if self.shader_manager else ""
                    active_idx = self.shader_manager.active_index if self.shader_manager else 0
                    total_count = self.shader_manager.shader_count if self.shader_manager else 0
                    last_err = self.shader_manager.last_error if self.shader_manager else None

                    self.hud.update_overlay(
                        self.window_width, self.window_height,
                        active_name,
                        active_idx,
                        total_count,
                        self.mask_mode, self.intensity, self.mirrored,
                        self.render_fps, self.render_ms,
                        cam_fps, cam_res, cam_backend,
                        tracking_state,
                        last_err,
                        extra_gestures_enabled=self.extra_gestures_enabled,
                        boost_mode=self.boost_mode,
                        frozen_shapes=self.frozen_shapes
                    )
                    self.hud.render(tracking_state, self.frozen_shapes)

                if self.window:
                    glfw.swap_buffers(self.window)
                    glfw.poll_events()

                t_frame_end = time.perf_counter()
                self.render_ms = (t_frame_end - t_frame_start) * 1000.0
                self._frame_count += 1
                self._total_frames_rendered += 1

                fps_elapsed = t_frame_end - self._fps_timer
                if fps_elapsed >= 1.0:
                    self.render_fps = self._frame_count / fps_elapsed
                    self._frame_count = 0
                    self._fps_timer = t_frame_end

                if self.max_frames > 0 and self._total_frames_rendered >= self.max_frames:
                    print(f"[Engine] Reached targat test frame count ({self.max_frames}). Exiting cleanly.")
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        print("[Engine] Shutting down....")
        if self.tracker:
            try:
                self.tracker.stop()
            except Exception:
                pass
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.hud:
            try:
                self.hud.cleanup()
            except Exception:
                pass
        if self.shader_manager:
            try:
                self.shader_manager.cleanup()
            except Exception:
                pass
        if self.cam_texture:
            try:
                glDeleteTextures(1, [self.cam_texture])
            except Exception:
                pass
            self.cam_texture = 0
        if self.window:
            try:
                glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None
        try:
            glfw.terminate()
        except Exception:
            pass
        print("[Engine] Shutdown complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="AirLens - GPU Shader Engine")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Preferred window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Preferred window height (default: 720)")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock video instead of physical camera")
    parser.add_argument("--no-vsync", action="store_true", help="Disable VSync for uncapped FPS benchmarking")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (useful for testing)")
    parser.add_argument("--headless", action="store_true", help="Run window invisible for automated tests")
    parser.add_argument("--boost", action="store_true", help="Enable tracking and gesture boost mode")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    app = VideoEffectsApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mock=args.mock,
        vsync=not args.no_vsync,
        max_frames=args.max_frames,
        boost=args.boost
    )
    app.run()

"""Integration test running VideoEffectsApp end-to-end."""

import unittest
from main import VideoEffectsApp


class TestIntegration(unittest.TestCase):

    def test_full_app_lifecycle(self):
        app = VideoEffectsApp(
            camera_index=0,
            width=640,
            height=480,
            mock=True,
            vsync=False,
            max_frames=30
        )
        app.run()
        self.assertEqual(app._total_frames_rendered, 30)
        self.assertIsNone(app.shader_manager.last_error)

    def test_hud_rendering_and_modes(self):
        app = VideoEffectsApp(
            camera_index=0,
            width=640,
            height=480,
            mock=True,
            vsync=False,
            max_frames=15
        )
        app.mask_mode = 2
        app.intensity = 1.5
        app.run()
        self.assertEqual(app._total_frames_rendered, 15)

    def test_gesture_events_lifecycle(self):
        from gesture import GestureEvent, GestureType

        app = VideoEffectsApp(
            camera_index=0,
            width=640,
            height=480,
            mock=True,
            vsync=False,
            max_frames=10
        )
        app.run()
        self.assertEqual(app._total_frames_rendered, 10)
        init_idx = app.shader_manager.active_index
        app._handle_gesture_event(GestureEvent(type=GestureType.SNAP))
        self.assertEqual(app.shader_manager.active_index, (init_idx + 1) % app.shader_manager.shader_count)
        self.assertIsNotNone(app.hud._gesture_toast)
        self.assertIn("SNAP", app.hud._gesture_toast["title"])

    def test_post_cleanup_hud_render_safety(self):
        """Verify that calling hud.render() after cleanup does not raise OpenGL GLError."""
        from tracker import TrackingData

        app = VideoEffectsApp(
            camera_index=0,
            width=640,
            height=480,
            mock=True,
            vsync=False,
            max_frames=5
        )
        app.run()
        try:
            app.hud.render(TrackingData(has_box=1.0))
        except Exception as e:
            self.fail(f"Post-cleanup hud.render() raised unexpected exception: {e}")

    def test_disable_extra_gestures_toggle(self):
        """Verify that toggling extra gestures disables recognition events while preserving app execution."""
        from gesture import GestureEvent, GestureType

        app = VideoEffectsApp(
            camera_index=0,
            width=640,
            height=480,
            mock=True,
            vsync=False,
            max_frames=5
        )
        app.run()

        self.assertTrue(app.extra_gestures_enabled)
        # Disable extra gestures
        app.toggle_extra_gestures()
        self.assertFalse(app.extra_gestures_enabled)
        self.assertFalse(app.tracker.extra_gestures_enabled)

        # Trigger snap event while extra gestures disabled -> shader index should NOT change
        idx_before = app.shader_manager.active_index
        app._handle_gesture_event(GestureEvent(type=GestureType.SNAP))
        self.assertEqual(app.shader_manager.active_index, idx_before)

        # Trigger continuous gesture -> intensity should NOT change
        int_before = app.intensity
        app._handle_continuous_gesture(GestureType.THUMBS_UP, dt=0.5)
        self.assertEqual(app.intensity, int_before)

        # Re-enable gestures
        app.toggle_extra_gestures()
        self.assertTrue(app.extra_gestures_enabled)
        self.assertTrue(app.tracker.extra_gestures_enabled)

        # Snap should work now
        app._handle_gesture_event(GestureEvent(type=GestureType.SNAP))
        self.assertEqual(app.shader_manager.active_index, (idx_before + 1) % app.shader_manager.shader_count)

    def test_gesture_button_click_detection(self):
        """Verify HUD gesture button hit-testing and hover tracking."""
        app = VideoEffectsApp(
            camera_index=0,
            width=640,
            height=480,
            mock=True,
            vsync=False,
            max_frames=5
        )
        app.run()

        bx0, by0, bx1, by1 = app.hud._btn_rect
        inside_x = (bx0 + bx1) / 2
        inside_y = (by0 + by1) / 2

        self.assertTrue(app.hud.is_gesture_button_clicked(inside_x, inside_y))
        self.assertFalse(app.hud.is_gesture_button_clicked(10, 10))
        self.assertFalse(app.hud.is_gesture_button_clicked(bx1 + 50, inside_y))

    def test_box_tracking_remains_active_when_gestures_disabled(self):
        """Verify that hand box tracking still updates when extra gestures are disabled."""
        from tracker import HandTracker

        tracker = HandTracker(app_camera := None)
        tracker.extra_gestures_enabled = False
        tracker._process_mock_gesture()
        state = tracker.get_state()

        self.assertEqual(state.has_box, 1.0)
        self.assertEqual(state.hands_count, 2)
        self.assertEqual(state.active_gesture, "box_only")
        self.assertTrue(state.box[2] > state.box[0])
        self.assertTrue(state.box[3] > state.box[1])

    def test_app_boost_mode_lifecycle(self):
        """Verify that running VideoEffectsApp with boost mode active executes cleanly."""
        app = VideoEffectsApp(
            camera_index=0,
            width=640,
            height=480,
            mock=True,
            vsync=False,
            max_frames=10,
            boost=True
        )
        self.assertTrue(app.boost_mode)
        app.run()
        self.assertEqual(app._total_frames_rendered, 10)
        self.assertTrue(app.tracker.boost_mode)
        self.assertTrue(app.tracker.get_state().boost_active)

    def test_high_dpi_mouse_mapping(self):
        """Verify High-DPI mapping scales cursor to framebuffer pixel space (BUG-03)."""
        from unittest.mock import patch, MagicMock
        import glfw

        app = VideoEffectsApp(mock=True, max_frames=1)
        app.window = MagicMock()
        with patch("glfw.get_window_size", return_value=(800, 600)), \
             patch("glfw.get_framebuffer_size", return_value=(1600, 1200)):
            sx, sy = app._get_dpi_scale()
            self.assertEqual(sx, 2.0)
            self.assertEqual(sy, 2.0)

            app.hud = MagicMock()
            app._on_mouse_move(app.window, 400, 300)
            # mouse_uv should be 0.5, 0.5 (relative to 800x600 window)
            self.assertAlmostEqual(app.mouse_uv[0], 0.5)
            self.assertAlmostEqual(app.mouse_uv[1], 0.5)
            # hud.set_mouse_pos should receive pixel coordinates (800, 600)
            app.hud.set_mouse_pos.assert_called_with(800.0, 600.0)

    def test_current_monitor_selection(self):
        """Verify _get_current_monitor selects the monitor containing the window (BUG-12)."""
        from unittest.mock import patch, MagicMock
        app = VideoEffectsApp(mock=True)
        app.window = MagicMock()

        m1 = MagicMock(name="Monitor1")
        m2 = MagicMock(name="Monitor2")
        mode1 = MagicMock()
        mode1.size.width = 1920
        mode1.size.height = 1080
        mode2 = MagicMock()
        mode2.size.width = 1920
        mode2.size.height = 1080

        # Window positioned on monitor 2 (x=2100, y=100)
        with patch("glfw.get_monitors", return_value=[m1, m2]), \
             patch("glfw.get_window_pos", return_value=(2100, 100)), \
             patch("glfw.get_window_size", return_value=(800, 600)), \
             patch("glfw.get_monitor_pos", side_effect=lambda m: (0, 0) if m == m1 else (1920, 0)), \
             patch("glfw.get_video_mode", side_effect=lambda m: mode1 if m == m1 else mode2):
            best = app._get_current_monitor()
            self.assertEqual(best, m2)

    def test_cleanup_guaranteed_on_exception(self):
        """Verify cleanup is invoked even if exception occurs during run (BUG-04)."""
        from unittest.mock import patch, MagicMock
        app = VideoEffectsApp(mock=True, max_frames=5)
        cleanup_called = False

        original_cleanup = app.cleanup
        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            original_cleanup()

        app.cleanup = mock_cleanup

        with patch.object(app, "init_glfw_and_gl", side_effect=RuntimeError("Simulated OpenGL failure")):
            with self.assertRaises(RuntimeError):
                app.run()

        self.assertTrue(cleanup_called)

    def test_camera_null_guard_on_key_x(self):
        """Verify pressing key X does not crash when camera is None (BUG-07)."""
        import glfw
        app = VideoEffectsApp(mock=True)
        app.camera = None
        # Should not raise AttributeError: 'NoneType' object has no attribute 'set_mirrored'
        app._on_key(None, glfw.KEY_X, 0, glfw.PRESS, 0)
        self.assertFalse(app.mirrored)

    def test_current_monitor_selection_window_none(self):
        """Verify _get_current_monitor safely returns primary monitor if window is None (BUG-12)."""
        from unittest.mock import patch, MagicMock
        app = VideoEffectsApp(mock=True)
        app.window = None
        mock_primary = MagicMock(name="PrimaryMonitor")
        with patch("glfw.get_primary_monitor", return_value=mock_primary):
            result = app._get_current_monitor()
            self.assertEqual(result, mock_primary)


if __name__ == "__main__":
    unittest.main()

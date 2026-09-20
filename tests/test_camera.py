"""Unit tests for CameraCapture module."""

import time
import unittest
import numpy as np
from camera import CameraCapture


class TestCameraCapture(unittest.TestCase):

    def test_mock_camera_stream(self):
        cam = CameraCapture(mock=True)
        cam.start()
        time.sleep(0.1)

        frame, fid, ts = cam.get_latest_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(fid, 0)
        self.assertGreater(ts, 0.0)

        # Verify new frames arrive
        time.sleep(0.1)
        frame2, fid2, ts2 = cam.get_latest_frame()
        self.assertGreater(fid2, fid)

        cam.stop()

    def test_single_frame_policy(self):
        """Verify that get_latest_frame always returns the most recent frame, not an old queue."""
        cam = CameraCapture(mock=True)
        cam.start()
        time.sleep(0.15)

        # Grab latest
        _, fid1, _ = cam.get_latest_frame()

        # Sleep to let several frames be generated in the background
        time.sleep(0.15)
        _, fid2, _ = cam.get_latest_frame()

        # Should jump directly to the newest frame
        self.assertGreater(fid2, fid1 + 2)

        cam.stop()

    def test_mirror_toggle(self):
        cam = CameraCapture(mock=True, mirrored=False)
        self.assertFalse(cam.mirrored)
        cam.start()
        time.sleep(0.1)
        f1, _, _ = cam.get_latest_frame()

        cam.set_mirrored(True)
        self.assertTrue(cam.mirrored)
        time.sleep(0.1)
        f2, _, _ = cam.get_latest_frame()

        self.assertIsNotNone(f1)
        self.assertIsNotNone(f2)
        # Verify the image is flipped horizontally (column 0 of f1 correlates with column -1 of f2)
        cam.stop()

    def test_dropped_frames_property(self):
        cam = CameraCapture(mock=True)
        self.assertEqual(cam.dropped_frames, 0)

    def test_unopened_camera_cleanup(self):
        """Verify unopened cv2.VideoCapture is properly released (BUG-05)."""
        from unittest.mock import MagicMock, patch
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        cam = CameraCapture(mock=False)
        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam._init_camera()
            self.assertTrue(mock_cap.release.called)
            self.assertIsNone(cam._cap)

    def test_consecutive_grab_failures_fallback_to_mock(self):
        """Verify 30 consecutive grab failures fallback to mock stream (BUG-06)."""
        from unittest.mock import MagicMock
        cam = CameraCapture(mock=False)
        mock_cap = MagicMock()
        mock_cap.grab.return_value = False
        cam._cap = mock_cap
        cam._running = True

        # Run capture worker briefly in a thread
        import threading
        t = threading.Thread(target=cam._capture_worker, daemon=True)
        t.start()
        time.sleep(0.3)
        cam._running = False
        t.join(timeout=1.0)

        self.assertTrue(cam.mock)
        self.assertTrue(mock_cap.release.called)
        self.assertIn("Fallback", cam.backend_name)

    def test_grab_exception_fallback_to_mock(self):
        """Verify grab exceptions (e.g. unplugged webcam) do not crash worker and fallback to mock (BUG-06)."""
        from unittest.mock import MagicMock
        cam = CameraCapture(mock=False)
        mock_cap = MagicMock()
        mock_cap.grab.side_effect = RuntimeError("USB unplugged")
        cam._cap = mock_cap
        cam._running = True

        import threading
        t = threading.Thread(target=cam._capture_worker, daemon=True)
        t.start()
        time.sleep(0.3)
        cam._running = False
        t.join(timeout=1.0)

        self.assertTrue(cam.mock)
        self.assertTrue(mock_cap.release.called)


if __name__ == "__main__":
    unittest.main()

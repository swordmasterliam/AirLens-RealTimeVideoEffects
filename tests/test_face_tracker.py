"""Unit tests for FaceTracker and WinkDetector."""

import time
import unittest
from face_tracker import WinkDetector, FaceTracker


class TestWinkDetector(unittest.TestCase):
    def setUp(self):
        self.detector = WinkDetector(cooldown=0.85, min_frames=2)

    def test_single_frame_does_not_trigger(self):
        """A single frame of wink closure should not fire; confirmation frame required."""
        triggered = self.detector.update(blink_left=0.85, blink_right=0.05, now=1.0)
        self.assertFalse(triggered, "Single frame wink should not trigger immediately")

    def test_two_consecutive_frames_triggers_left_wink(self):
        """Two consecutive frames of left wink should trigger a wink event."""
        res1 = self.detector.update(blink_left=0.85, blink_right=0.05, now=1.0)
        res2 = self.detector.update(blink_left=0.85, blink_right=0.05, now=1.033)
        self.assertFalse(res1)
        self.assertTrue(res2, "Second consecutive frame of left wink should trigger")

    def test_two_consecutive_frames_triggers_right_wink(self):
        """Two consecutive frames of right wink should trigger a wink event."""
        res1 = self.detector.update(blink_left=0.05, blink_right=0.85, now=1.0)
        res2 = self.detector.update(blink_left=0.05, blink_right=0.85, now=1.033)
        self.assertFalse(res1)
        self.assertTrue(res2, "Second consecutive frame of right wink should trigger")

    def test_bilateral_blink_rejection(self):
        """Both eyes blinking simultaneously must be rejected as a normal blink, not a wink."""
        # Both eyes fully closed
        res1 = self.detector.update(blink_left=0.90, blink_right=0.90, now=1.0)
        res2 = self.detector.update(blink_left=0.90, blink_right=0.90, now=1.033)
        self.assertFalse(res1)
        self.assertFalse(res2, "Bilateral blink must not trigger wink")

        # Both eyes moderately closed (> 0.38)
        res3 = self.detector.update(blink_left=0.50, blink_right=0.45, now=2.0)
        res4 = self.detector.update(blink_left=0.50, blink_right=0.45, now=2.033)
        self.assertFalse(res3)
        self.assertFalse(res4, "Moderate bilateral squint/blink must not trigger wink")

    def test_low_differential_rejection(self):
        """Wink requires clear asymmetry (differential >= 0.28)."""
        res1 = self.detector.update(blink_left=0.46, blink_right=0.25, now=1.0)
        res2 = self.detector.update(blink_left=0.46, blink_right=0.25, now=1.033)
        self.assertFalse(res1)
        self.assertFalse(res2, "Small differential (0.21 < 0.28) must not trigger wink")

    def test_cooldown_enforcement(self):
        """Subsequent winks within cooldown period must be ignored."""
        # First wink at t=1.0 -> triggers at 1.033
        self.detector.update(blink_left=0.9, blink_right=0.05, now=1.0)
        trig = self.detector.update(blink_left=0.9, blink_right=0.05, now=1.033)
        self.assertTrue(trig)

        # Attempt second wink at t=1.5 (within 0.85s cooldown)
        res1 = self.detector.update(blink_left=0.9, blink_right=0.05, now=1.5)
        res2 = self.detector.update(blink_left=0.9, blink_right=0.05, now=1.533)
        self.assertFalse(res1)
        self.assertFalse(res2, "Wink within cooldown period must be suppressed")

        # Third wink at t=2.0 (> 0.85s after first trigger at 1.033)
        res3 = self.detector.update(blink_left=0.9, blink_right=0.05, now=2.0)
        res4 = self.detector.update(blink_left=0.9, blink_right=0.05, now=2.033)
        self.assertFalse(res3)
        self.assertTrue(res4, "Wink after cooldown period expires must trigger successfully")


class TestFaceTracker(unittest.TestCase):
    def test_tracker_initialization(self):
        """FaceTracker should initialize and have WinkDetector configured."""
        tracker = FaceTracker(cooldown=0.75)
        self.assertIsNotNone(tracker.wink_detector)
        self.assertEqual(tracker.wink_detector.cooldown, 0.75)
        self.assertFalse(tracker.face_present)
        self.assertEqual(tracker.blink_left, 0.0)
        self.assertEqual(tracker.blink_right, 0.0)
        tracker.close()


if __name__ == "__main__":
    unittest.main()

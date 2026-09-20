"""Unit tests for HandTracker and smoothing module."""

import unittest
from types import SimpleNamespace
import numpy as np
from tracker import AdaptiveSmoother, HandTracker, TrackingData
from camera import CameraCapture


class TestAdaptiveSmoother(unittest.TestCase):

    def test_smoothing(self):
        smoother = AdaptiveSmoother(min_alpha=0.3, max_alpha=0.8)
        p1 = np.array([0.0, 0.0, 1.0, 1.0])
        s1 = smoother.update(p1)
        np.testing.assert_array_equal(s1, p1)

        p2 = np.array([0.02, 0.02, 1.02, 1.02])
        s2 = smoother.update(p2)
        self.assertTrue(np.all(s2 < p2))
        self.assertTrue(np.all(s2 > p1))

        smoother.reset()
        self.assertIsNone(smoother.prev_val)


class TestHandTrackerLogic(unittest.TestCase):

    def test_process_landmarks_2_hands(self):
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)

        def make_landmark(x, y):
            return SimpleNamespace(x=x, y=y, z=0.0)

        hand1 = [make_landmark(0.0, 0.0)] * 21
        hand1[4] = make_landmark(0.2, 0.3)
        hand1[8] = make_landmark(0.3, 0.2)

        hand2 = [make_landmark(0.0, 0.0)] * 21
        hand2[4] = make_landmark(0.7, 0.6)
        hand2[8] = make_landmark(0.8, 0.5)

        mock_result = SimpleNamespace(hand_landmarks=[hand1, hand2])
        tracker._process_landmarks(mock_result, 5.0)

        state = tracker.get_state()
        self.assertEqual(state.hands_count, 2)
        self.assertEqual(state.has_box, 1.0)
        self.assertLessEqual(state.box[0], 0.2)
        self.assertGreaterEqual(state.box[2], 0.8)

    def test_process_landmarks_1_hand(self):
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)

        def make_landmark(x, y):
            return SimpleNamespace(x=x, y=y, z=0.0)

        hand1 = [make_landmark(0.0, 0.0)] * 21
        hand1[4] = make_landmark(0.4, 0.4)
        hand1[8] = make_landmark(0.6, 0.6)

        mock_result = SimpleNamespace(hand_landmarks=[hand1])
        tracker._process_landmarks(mock_result, 4.0)

        state = tracker.get_state()
        self.assertEqual(state.hands_count, 1)
        self.assertEqual(state.has_box, 1.0)
        self.assertLess(state.box[0], 0.4)
        self.assertGreater(state.box[2], 0.6)

    def test_process_landmarks_0_hands_fade(self):
        cam = CameraCapture(mock=False)
        tracker = HandTracker(cam)
        tracker._state = TrackingData(has_box=1.0)

        tracker._process_landmarks(None, 2.0)
        state = tracker.get_state()
        self.assertLess(state.has_box, 1.0)
        self.assertIsNone(tracker._f1_smoother.prev_val)
        self.assertIsNone(tracker._f2_smoother.prev_val)

    def test_deterministic_hand_sorting(self):
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)

        def make_landmark(x, y):
            return SimpleNamespace(x=x, y=y, z=0.0)

        right_hand = [make_landmark(0.0, 0.0)] * 21
        right_hand[4] = make_landmark(0.75, 0.6)
        right_hand[8] = make_landmark(0.85, 0.5)

        left_hand = [make_landmark(0.0, 0.0)] * 21
        left_hand[4] = make_landmark(0.2, 0.3)
        left_hand[8] = make_landmark(0.3, 0.2)

        mock_result = SimpleNamespace(hand_landmarks=[right_hand, left_hand])
        tracker._process_landmarks(mock_result, 4.0)

        state = tracker.get_state()
        self.assertAlmostEqual(state.thumb1[0], 0.2, places=2)
        self.assertAlmostEqual(state.index1[0], 0.3, places=2)
        self.assertAlmostEqual(state.thumb2[0], 0.75, places=2)
        self.assertAlmostEqual(state.index2[0], 0.85, places=2)

    def test_mock_gesture_simulation(self):
        """Verify that mock gesture produces an active two-hand box in mock mode."""
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)
        tracker._process_mock_gesture()

        state = tracker.get_state()
        self.assertEqual(state.has_box, 1.0)
        self.assertEqual(state.hands_count, 2)
        self.assertLess(state.box[0], state.box[2])
        self.assertLess(state.box[1], state.box[3])
        self.assertGreater(state.thumb1[0], 0.0)
        self.assertGreater(state.index2[0], 0.0)


    def test_hand_tracker_persistence(self):
        """Verify that hand identity is preserved across frames when list order changes."""
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)

        def make_landmark(x, y):
            return SimpleNamespace(x=x, y=y, z=0.0)

        h_left = [make_landmark(0.0, 0.0)] * 21
        h_left[4] = make_landmark(0.25, 0.3)
        h_left[8] = make_landmark(0.30, 0.2)

        h_right = [make_landmark(0.0, 0.0)] * 21
        h_right[4] = make_landmark(0.70, 0.6)
        h_right[8] = make_landmark(0.75, 0.5)

        # Frame 1: initial detection sorts left to right
        tracker._process_landmarks(SimpleNamespace(hand_landmarks=[h_right, h_left]), 3.0)
        s1 = tracker.get_state()
        self.assertAlmostEqual(s1.thumb1[0], 0.25, places=2)
        self.assertAlmostEqual(s1.thumb2[0], 0.70, places=2)

        # Frame 2: inputs arrive in reverse order or hands move slightly
        h_left_moved = [make_landmark(0.0, 0.0)] * 21
        h_left_moved[4] = make_landmark(0.26, 0.3)
        h_left_moved[8] = make_landmark(0.31, 0.2)

        h_right_moved = [make_landmark(0.0, 0.0)] * 21
        h_right_moved[4] = make_landmark(0.69, 0.6)
        h_right_moved[8] = make_landmark(0.74, 0.5)

        # Pass in reversed order
        tracker._process_landmarks(SimpleNamespace(hand_landmarks=[h_right_moved, h_left_moved]), 3.0)
        s2 = tracker.get_state()
        # Hand 1 must remain the left hand (~0.26) and Hand 2 remain the right hand (~0.69)
        self.assertAlmostEqual(s2.thumb1[0], 0.26, places=1)
        self.assertAlmostEqual(s2.thumb2[0], 0.69, places=1)

    def test_adaptive_smoother_boost(self):
        """Verify boost mode increases velocity scale and max alpha for snappier tracking."""
        smoother = AdaptiveSmoother(min_alpha=0.3, max_alpha=0.8, velocity_scale=5.0)
        smoother.update(np.array([0.0, 0.0]))
        res_normal = smoother.update(np.array([0.1, 0.1]))

        smoother_boost = AdaptiveSmoother(min_alpha=0.3, max_alpha=0.8, velocity_scale=5.0)
        smoother_boost.set_boost(True)
        smoother_boost.update(np.array([0.0, 0.0]))
        res_boost = smoother_boost.update(np.array([0.1, 0.1]))

        # In boost mode, alpha is higher so smoothed value is closer to the target (0.1)
        self.assertGreater(res_boost[0], res_normal[0])

    def test_tracker_boost_mode(self):
        """Verify tracker set_boost propagates state and updates TrackingData."""
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)
        self.assertFalse(tracker.boost_mode)

        tracker.set_boost(True)
        self.assertTrue(tracker.boost_mode)
        self.assertTrue(tracker._box_smoother.boost_mode)
        self.assertTrue(tracker.gesture_recognizer.boost_mode)

        tracker._process_mock_gesture()
        state = tracker.get_state()
        self.assertTrue(state.boost_active)

    def test_tracker_init_with_boost(self):
        """Verify HandTracker can be directly initialized with boost=True."""
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam, boost=True)
        self.assertTrue(tracker.boost_mode)
        self.assertTrue(tracker._box_smoother.boost_mode)
        self.assertTrue(tracker.gesture_recognizer.boost_mode)

    def test_hand_transition_two_to_one_resets_smoother_to_prevent_drag(self):
        """When hand 0 vanishes and hand 1 remains, smoother is reset so thumb1 doesn't drag across screen."""
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)

        def make_landmark(x, y):
            return SimpleNamespace(x=x, y=y, z=0.0)

        h0 = [make_landmark(0.0, 0.0)] * 21
        h0[4] = make_landmark(0.20, 0.3)
        h0[8] = make_landmark(0.22, 0.2)

        h1 = [make_landmark(0.0, 0.0)] * 21
        h1[4] = make_landmark(0.80, 0.6)
        h1[8] = make_landmark(0.82, 0.5)

        # Frame 1: both hands
        tracker._process_landmarks(SimpleNamespace(hand_landmarks=[h0, h1]), 3.0)
        s1 = tracker.get_state()
        self.assertAlmostEqual(s1.thumb1[0], 0.20, places=1)
        self.assertAlmostEqual(s1.thumb2[0], 0.80, places=1)

        # Frame 2: hand 0 disappears, only hand 1 (at 0.80) remains
        tracker._process_landmarks(SimpleNamespace(hand_landmarks=[h1]), 3.0)
        s2 = tracker.get_state()
        # thumb1 should immediately snap to ~0.80, NOT lag at ~0.40 due to smoothing from 0.20
        self.assertGreater(s2.thumb1[0], 0.70)
        self.assertEqual(s2.hands_count, 1)

    def test_adaptive_smoother_nan_inf_protection(self):
        """Verify AdaptiveSmoother ignores NaN/Inf values and prevents corruption (BUG-11)."""
        smoother = AdaptiveSmoother()
        # Case 1: uninitialized with NaN
        res1 = smoother.update(np.array([np.nan, 1.0]))
        self.assertTrue(np.all(np.isfinite(res1)))
        self.assertIsNone(smoother.prev_val)

        # Initialize with valid value
        valid = np.array([0.5, 0.5])
        smoother.update(valid)

        # Case 2: update with Inf/NaN retains previous valid value
        res_nan = smoother.update(np.array([np.nan, np.inf]))
        np.testing.assert_array_equal(res_nan, valid)
        np.testing.assert_array_equal(smoother.prev_val, valid)

    def test_process_landmarks_short_or_malformed_hand(self):
        """Verify tracker ignores hands with < 21 landmarks without crashing (BUG-01)."""
        cam = CameraCapture(mock=False)
        tracker = HandTracker(cam)

        def make_landmark(x, y):
            return SimpleNamespace(x=x, y=y, z=0.0)

        short_hand = [make_landmark(0.1, 0.2)] * 10  # Only 10 landmarks
        mock_result = SimpleNamespace(hand_landmarks=[short_hand])

        # Should not raise IndexError
        tracker._process_landmarks(mock_result, 2.0)
        state = tracker.get_state()
        self.assertEqual(state.hands_count, 0)

    def test_process_landmarks_none_hand_landmarks(self):
        """Verify tracker handles result with hand_landmarks=None without TypeError (BUG-01)."""
        cam = CameraCapture(mock=False)
        tracker = HandTracker(cam)
        mock_result = SimpleNamespace(hand_landmarks=None)

        # Should not raise TypeError: 'NoneType' object is not iterable
        tracker._process_landmarks(mock_result, 2.0)
        state = tracker.get_state()
        self.assertEqual(state.hands_count, 0)

    def test_process_landmarks_three_hands_tracking(self):
        """Verify tracker stably tracks when 3 or more hands are detected (BUG-01)."""
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)

        def make_landmark(x, y):
            return SimpleNamespace(x=x, y=y, z=0.0)

        h1 = [make_landmark(0.0, 0.0)] * 21
        h1[4] = make_landmark(0.2, 0.3)
        h1[8] = make_landmark(0.3, 0.2)

        h2 = [make_landmark(0.0, 0.0)] * 21
        h2[4] = make_landmark(0.7, 0.6)
        h2[8] = make_landmark(0.8, 0.5)

        h3 = [make_landmark(0.0, 0.0)] * 21
        h3[4] = make_landmark(0.5, 0.5)
        h3[8] = make_landmark(0.5, 0.5)

        mock_result = SimpleNamespace(hand_landmarks=[h1, h2, h3])
        tracker._process_landmarks(mock_result, 5.0)

        state = tracker.get_state()
        self.assertEqual(state.hands_count, 3)
        self.assertEqual(state.has_box, 1.0)
        # Should track top 2 hands (thumb1 and thumb2 both populated)
        self.assertAlmostEqual(state.thumb1[0], 0.2, places=1)
        self.assertAlmostEqual(state.thumb2[0], 0.7, places=1)

    def test_adaptive_smoother_python_list_support(self):
        """Verify AdaptiveSmoother accepts list inputs as well as numpy arrays."""
        smoother = AdaptiveSmoother()
        res = smoother.update([0.5, 0.5])
        self.assertIsInstance(res, np.ndarray)
        self.assertAlmostEqual(res[0], 0.5)


if __name__ == "__main__":
    unittest.main()

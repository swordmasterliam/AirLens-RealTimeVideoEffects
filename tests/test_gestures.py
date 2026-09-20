"""Unit and integration tests for hand gesture recognition engine."""

import time
import unittest
import threading
from types import SimpleNamespace
from gesture import (
    GestureType,
    GestureEvent,
    GestureRecognizer,
    SnapDetector,
    point_dist,
    get_hand_scale,
    is_finger_extended,
    is_finger_folded,
    is_thumb_extended,
    is_thumb_folded,
    check_peace,
    check_thumbs_up,
    check_thumbs_down,
    check_fist,
    check_open_palm,
    check_boost,
)
from tracker import HandTracker, TrackingData
from camera import CameraCapture


def make_lm(x, y, z=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def create_base_hand(wrist=(0.5, 0.7), middle_mcp=(0.5, 0.5)):
    """Create 21 landmark hand template with stable wrist and metacarpal bases."""
    hand = [make_lm(0.0, 0.0)] * 21
    hand[0] = make_lm(wrist[0], wrist[1])
    hand[1] = make_lm(wrist[0] - 0.05, wrist[1] - 0.05)  # Thumb CMC
    hand[2] = make_lm(wrist[0] - 0.07, wrist[1] - 0.10)  # Thumb MCP
    hand[3] = make_lm(wrist[0] - 0.10, wrist[1] - 0.18)  # Thumb IP
    hand[4] = make_lm(wrist[0] - 0.12, wrist[1] - 0.25)  # Thumb TIP

    hand[5] = make_lm(0.45, 0.52)   # Index MCP
    hand[6] = make_lm(0.44, 0.42)   # Index PIP
    hand[7] = make_lm(0.43, 0.35)   # Index DIP
    hand[8] = make_lm(0.42, 0.25)   # Index TIP

    hand[9] = make_lm(middle_mcp[0], middle_mcp[1])  # Middle MCP
    hand[10] = make_lm(0.50, 0.38)  # Middle PIP
    hand[11] = make_lm(0.50, 0.30)  # Middle DIP
    hand[12] = make_lm(0.50, 0.22)  # Middle TIP

    hand[13] = make_lm(0.55, 0.52)  # Ring MCP
    hand[14] = make_lm(0.56, 0.40)  # Ring PIP
    hand[15] = make_lm(0.56, 0.32)  # Ring DIP
    hand[16] = make_lm(0.56, 0.25)  # Ring TIP

    hand[17] = make_lm(0.58, 0.55)  # Pinky MCP
    hand[18] = make_lm(0.59, 0.45)  # Pinky PIP
    hand[19] = make_lm(0.60, 0.38)  # Pinky DIP
    hand[20] = make_lm(0.60, 0.30)  # Pinky TIP
    return hand


def create_open_palm_hand():
    """All 5 fingers extended outward."""
    return create_base_hand()


def create_fist_hand():
    """All 5 fingers curled into palm."""
    hand = create_base_hand()
    # Curl Index (tip near MCP, closer to wrist than PIP)
    hand[6] = make_lm(0.45, 0.46)
    hand[8] = make_lm(0.45, 0.54)
    # Curl Middle
    hand[10] = make_lm(0.50, 0.45)
    hand[12] = make_lm(0.50, 0.53)
    # Curl Ring
    hand[14] = make_lm(0.55, 0.46)
    hand[16] = make_lm(0.55, 0.54)
    # Curl Pinky
    hand[18] = make_lm(0.58, 0.48)
    hand[20] = make_lm(0.58, 0.56)
    # Curl Thumb over fingers
    hand[4] = make_lm(0.48, 0.52)
    return hand


def create_peace_hand():
    """Index & Middle extended, Ring & Pinky curled."""
    hand = create_fist_hand()
    # Extend Index
    hand[6] = make_lm(0.43, 0.40)
    hand[8] = make_lm(0.42, 0.25)
    # Extend Middle
    hand[10] = make_lm(0.52, 0.38)
    hand[12] = make_lm(0.54, 0.23)
    return hand


def create_thumbs_up_hand():
    """Fist with thumb pointing upwards (-y)."""
    hand = create_fist_hand()
    hand[2] = make_lm(0.45, 0.60)
    hand[3] = make_lm(0.45, 0.50)
    hand[4] = make_lm(0.45, 0.36)  # Pointing up
    return hand


def create_thumbs_down_hand():
    """Fist with thumb pointing downwards (+y)."""
    hand = create_fist_hand()
    hand[0] = make_lm(0.5, 0.40)   # Wrist high
    hand[9] = make_lm(0.5, 0.60)   # Metacarpals lower
    hand[2] = make_lm(0.45, 0.50)
    hand[3] = make_lm(0.45, 0.60)
    hand[4] = make_lm(0.45, 0.74)  # Pointing down (lower than wrist)
    return hand


def create_boost_hand():
    """Rock-on / horns: index & pinky extended, middle & ring curled."""
    hand = create_fist_hand()
    # Extend Index
    hand[6] = make_lm(0.43, 0.40)
    hand[8] = make_lm(0.42, 0.25)
    # Extend Pinky
    hand[18] = make_lm(0.59, 0.45)
    hand[20] = make_lm(0.60, 0.30)
    return hand


class TestGestureGeometry(unittest.TestCase):

    def test_point_dist(self):
        p1 = make_lm(0.1, 0.2)
        p2 = make_lm(0.4, 0.6)
        # 3-4-5 triangle
        self.assertAlmostEqual(point_dist(p1, p2), 0.5, places=5)

    def test_hand_scale(self):
        hand = create_base_hand(wrist=(0.5, 0.7), middle_mcp=(0.5, 0.5))
        scale = get_hand_scale(hand)
        self.assertAlmostEqual(scale, 0.20, places=4)

    def test_finger_extension_and_folded(self):
        open_hand = create_open_palm_hand()
        scale = get_hand_scale(open_hand)
        self.assertTrue(is_finger_extended(open_hand, 8, 6, 5, scale))
        self.assertTrue(is_finger_extended(open_hand, 12, 10, 9, scale))
        self.assertFalse(is_finger_folded(open_hand, 8, 6, 5, scale))

        fist_hand = create_fist_hand()
        scale_fist = get_hand_scale(fist_hand)
        self.assertTrue(is_finger_folded(fist_hand, 8, 6, 5, scale_fist))
        self.assertFalse(is_finger_extended(fist_hand, 8, 6, 5, scale_fist))


class TestStaticGestures(unittest.TestCase):

    def test_peace_sign_recognition(self):
        peace_hand = create_peace_hand()
        scale = get_hand_scale(peace_hand)
        self.assertTrue(check_peace(peace_hand, scale))
        self.assertFalse(check_fist(peace_hand, scale))
        self.assertFalse(check_open_palm(peace_hand, scale))
        self.assertFalse(check_thumbs_up(peace_hand, scale))

    def test_closed_fist_recognition(self):
        fist_hand = create_fist_hand()
        scale = get_hand_scale(fist_hand)
        self.assertTrue(check_fist(fist_hand, scale))
        self.assertFalse(check_peace(fist_hand, scale))
        self.assertFalse(check_open_palm(fist_hand, scale))
        self.assertFalse(check_thumbs_up(fist_hand, scale))

    def test_open_palm_recognition(self):
        open_hand = create_open_palm_hand()
        scale = get_hand_scale(open_hand)
        self.assertTrue(check_open_palm(open_hand, scale))
        self.assertFalse(check_fist(open_hand, scale))
        self.assertFalse(check_peace(open_hand, scale))

    def test_thumbs_up_recognition(self):
        t_up = create_thumbs_up_hand()
        scale = get_hand_scale(t_up)
        self.assertTrue(check_thumbs_up(t_up, scale))
        self.assertFalse(check_thumbs_down(t_up, scale))
        self.assertFalse(check_fist(t_up, scale))

    def test_thumbs_down_recognition(self):
        t_down = create_thumbs_down_hand()
        scale = get_hand_scale(t_down)
        self.assertTrue(check_thumbs_down(t_down, scale))
        self.assertFalse(check_thumbs_up(t_down, scale))
        self.assertFalse(check_fist(t_down, scale))

    def test_thumbs_down_realistic_hand_position(self):
        """Verify thumbs down recognition when arm enters from bottom of frame (wrist below hand)."""
        hand = create_fist_hand()
        hand[0] = make_lm(0.5, 0.70)   # Wrist near bottom
        hand[9] = make_lm(0.5, 0.50)   # Metacarpals higher
        scale = get_hand_scale(hand)
        hand[2] = make_lm(0.45, 0.50)  # Thumb MCP
        hand[3] = make_lm(0.45, 0.58)  # Thumb IP
        hand[4] = make_lm(0.45, 0.65)  # Thumb tip pointing downwards
        self.assertTrue(check_thumbs_down(hand, scale))
        self.assertFalse(check_thumbs_up(hand, scale))
        self.assertFalse(check_fist(hand, scale))

    def test_thumbs_up_scale_invariance_small_hand(self):
        """Verify scale invariance when hand is far from camera (scale = 0.08)."""
        hand = create_fist_hand()
        hand[0] = make_lm(0.5, 0.58)
        hand[9] = make_lm(0.5, 0.50)
        scale = get_hand_scale(hand)
        self.assertAlmostEqual(scale, 0.08, places=3)
        hand[2] = make_lm(0.48, 0.54)
        hand[3] = make_lm(0.48, 0.525)
        hand[4] = make_lm(0.48, 0.51)
        self.assertTrue(check_thumbs_up(hand, scale))

    def test_peace_sign_rejects_three_fingers(self):
        """Peace sign should reject hands where thumb is also extended outward."""
        peace_hand = create_peace_hand()
        scale = get_hand_scale(peace_hand)
        self.assertTrue(check_peace(peace_hand, scale))
        # Extend thumb outward
        peace_hand[4] = make_lm(0.30, 0.50)
        self.assertFalse(check_peace(peace_hand, scale))

    def test_pinch_does_not_trigger_fist(self):
        """When thumb and middle finger are compressed in snap prep, check_fist must be False."""
        snap_prep_hand = create_fist_hand()
        scale = get_hand_scale(snap_prep_hand)
        # Pinch thumb (4) and middle (12)
        snap_prep_hand[4] = make_lm(0.49, 0.45)
        snap_prep_hand[12] = make_lm(0.50, 0.45)
        self.assertFalse(check_fist(snap_prep_hand, scale))

    def test_malformed_hand_handling(self):
        """Verify robust handling of empty or truncated landmarks."""
        short_hand = [make_lm(0.0, 0.0)] * 10
        self.assertFalse(check_peace(short_hand, 0.2))
        self.assertFalse(check_fist(short_hand, 0.2))
        self.assertFalse(check_thumbs_up(short_hand, 0.2))
        self.assertFalse(check_thumbs_down(short_hand, 0.2))
        self.assertFalse(check_open_palm(short_hand, 0.2))
        self.assertFalse(check_boost(short_hand, 0.2))

    def test_boost_gesture_recognition(self):
        """Verify rock-on/horns gesture is recognized and isolated from other gestures."""
        b_hand = create_boost_hand()
        scale = get_hand_scale(b_hand)
        self.assertTrue(check_boost(b_hand, scale))
        self.assertFalse(check_peace(b_hand, scale))
        self.assertFalse(check_fist(b_hand, scale))
        self.assertFalse(check_open_palm(b_hand, scale))
        self.assertFalse(check_thumbs_up(b_hand, scale))

    def test_fist_rejects_extended_thumb(self):
        """Fist gesture must strictly reject hands where thumb is extended (isolation)."""
        hand = create_fist_hand()
        scale = get_hand_scale(hand)
        self.assertTrue(check_fist(hand, scale))

        # Extend thumb outward horizontally (hitchhiker / sideways thumb)
        hand[2] = make_lm(0.38, 0.60)
        hand[3] = make_lm(0.30, 0.60)
        hand[4] = make_lm(0.22, 0.60)
        self.assertFalse(check_fist(hand, scale), "Fist must reject extended thumb")

    def test_boost_gesture_with_extended_thumb(self):
        """Verify boost gesture is recognized with extended thumb (Spider-man/rock-on sign)."""
        b_hand = create_boost_hand()
        # Extend thumb outward
        b_hand[2] = make_lm(0.40, 0.60)
        b_hand[3] = make_lm(0.35, 0.55)
        b_hand[4] = make_lm(0.30, 0.50)
        scale = get_hand_scale(b_hand)
        self.assertTrue(check_boost(b_hand, scale))
        self.assertFalse(check_peace(b_hand, scale))
        self.assertFalse(check_open_palm(b_hand, scale))
        self.assertFalse(check_fist(b_hand, scale))


class TestSnapDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SnapDetector(cooldown=0.5)

    def test_successful_snap(self):
        hand = create_base_hand()
        scale = get_hand_scale(hand)

        hand[4] = make_lm(0.35, 0.5)
        hand[12] = make_lm(0.50, 0.22)
        fired = self.detector.update(hand, scale, now=1.0)
        self.assertFalse(fired)
        self.assertEqual(self.detector.state, "IDLE")

        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        fired = self.detector.update(hand, scale, now=1.10)
        self.assertFalse(fired)
        self.assertEqual(self.detector.state, "COMPRESSED")

        hand[4] = make_lm(0.40, 0.40)
        hand[12] = make_lm(0.55, 0.56)
        fired = self.detector.update(hand, scale, now=1.16)
        self.assertTrue(fired, "Expected snap event to be detected")
        self.assertEqual(self.detector.state, "IDLE")

    def test_slow_release_does_not_snap(self):
        """Gradual separation taking > 0.4s should NOT be recognized as snap."""
        hand = create_base_hand()
        scale = get_hand_scale(hand)

        # Compression
        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        self.detector.update(hand, scale, now=1.0)
        self.assertEqual(self.detector.state, "COMPRESSED")

        # Separation delayed by 0.5s
        hand[4] = make_lm(0.35, 0.35)
        hand[12] = make_lm(0.55, 0.55)
        fired = self.detector.update(hand, scale, now=1.55)
        self.assertFalse(fired, "Slow release should not trigger snap")

    def test_snap_cooldown_prevents_double_fire(self):
        """Cooldown period blocks rapid spurious multi-firing."""
        hand = create_base_hand()
        scale = get_hand_scale(hand)

        # Snap 1
        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        self.detector.update(hand, scale, now=1.0)
        hand[4] = make_lm(0.35, 0.35)
        hand[12] = make_lm(0.55, 0.55)
        fired1 = self.detector.update(hand, scale, now=1.06)
        self.assertTrue(fired1)

        # Immediate attempt during cooldown (0.2s later)
        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        self.detector.update(hand, scale, now=1.20)
        hand[4] = make_lm(0.35, 0.35)
        hand[12] = make_lm(0.55, 0.55)
        fired2 = self.detector.update(hand, scale, now=1.26)
        self.assertFalse(fired2, "Snap within cooldown must be suppressed")

        # Attempt after cooldown expires (0.6s later)
        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        self.detector.update(hand, scale, now=1.70)
        hand[4] = make_lm(0.35, 0.35)
        hand[12] = make_lm(0.55, 0.55)
        fired3 = self.detector.update(hand, scale, now=1.76)
        self.assertTrue(fired3, "Snap after cooldown should trigger successfully")

    def test_snap_after_prolonged_pinch(self):
        hand = create_base_hand()
        scale = get_hand_scale(hand)

        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        self.detector.update(hand, scale, now=1.0)
        self.assertEqual(self.detector.state, "COMPRESSED")

        self.detector.update(hand, scale, now=1.5)
        self.detector.update(hand, scale, now=2.0)
        self.detector.update(hand, scale, now=2.2)
        self.assertEqual(self.detector.state, "COMPRESSED")

        hand[4] = make_lm(0.40, 0.40)
        hand[12] = make_lm(0.55, 0.56)
        fired = self.detector.update(hand, scale, now=2.25)
        self.assertTrue(fired, "Snap must trigger even after prolonged compression")

    def test_snap_with_curled_index_finger(self):
        """Snapping middle finger while index finger is curled near thumb must succeed."""
        hand = create_base_hand()
        scale = get_hand_scale(hand)

        # Index finger curled near thumb (d_index ~ 0.28)
        hand[8] = make_lm(0.45, 0.45)

        # Middle finger pinched with thumb
        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        self.detector.update(hand, scale, now=1.0)
        self.assertEqual(self.detector.state, "COMPRESSED")

        # Middle finger snaps away, index finger stays curled
        hand[4] = make_lm(0.40, 0.40)
        hand[12] = make_lm(0.55, 0.56)
        fired = self.detector.update(hand, scale, now=1.05)
        self.assertTrue(fired, "Snap must trigger independently of index finger position")

    def test_snap_ending_in_palm_fist_pose(self):
        """Snap release where finger impacts palm must still register despite fist-like posture."""
        hand = create_base_hand()
        scale = get_hand_scale(hand)

        # Compress
        hand[4] = make_lm(0.49, 0.43)
        hand[12] = make_lm(0.50, 0.43)
        self.detector.update(hand, scale, now=1.0)
        self.assertEqual(self.detector.state, "COMPRESSED")

        # Snap into curled fist pose
        fist_hand = create_fist_hand()
        fist_hand[4] = make_lm(0.40, 0.40)
        fired = self.detector.update(fist_hand, scale, now=1.06)
        self.assertTrue(fired, "Snap finishing in palm fist pose must be recognized")


class TestGestureRecognizerEngine(unittest.TestCase):

    def setUp(self):
        self.rec = GestureRecognizer(cooldown=0.6)

    def test_peace_sign_debouncing_and_event_queue(self):
        """Verify peace sign requires 2 stable frames and triggers once until released."""
        peace_hand = create_peace_hand()

        # Frame 1: candidate detected, but needs 2 frames of hysteresis
        self.rec.update([peace_hand], now=1.0)
        events = self.rec.poll_events()
        self.assertEqual(len(events), 0)

        # Frame 2: confirmed! Event emitted
        self.rec.update([peace_hand], now=1.033)
        events = self.rec.poll_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, GestureType.PEACE)

        # Frame 3: held steady - should not emit duplicate event
        self.rec.update([peace_hand], now=1.066)
        events = self.rec.poll_events()
        self.assertEqual(len(events), 0)

        # Release to open palm
        open_hand = create_open_palm_hand()
        self.rec.update([open_hand], now=1.10)
        self.rec.poll_events()

        # Show peace sign again after cooldown -> should emit again
        self.rec.update([peace_hand], now=1.80)
        self.rec.update([peace_hand], now=1.833)
        events2 = self.rec.poll_events()
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0].type, GestureType.PEACE)

    def test_continuous_gesture_tracking(self):
        """Verify Thumbs Up sets continuous state while held and clears when released."""
        t_up = create_thumbs_up_hand()

        self.assertIsNone(self.rec.get_continuous_gesture())

        # Frame 1
        self.rec.update([t_up], now=1.0)
        # Frame 2 (confirmed continuous)
        self.rec.update([t_up], now=1.033)
        self.assertEqual(self.rec.get_continuous_gesture(), GestureType.THUMBS_UP)

        # Open palm releases continuous gesture
        open_hand = create_open_palm_hand()
        self.rec.update([open_hand], now=1.066)
        self.rec.update([open_hand], now=1.10)
        self.assertIsNone(self.rec.get_continuous_gesture())

    def test_thread_safety_stress(self):
        """Concurrent background tracking updates and foreground event polling."""
        rec = GestureRecognizer(cooldown=0.05)
        stop_event = threading.Event()
        errors = []

        def background_updater():
            hand = create_peace_hand()
            t = 0.0
            while not stop_event.is_set():
                try:
                    rec.update([hand], now=t)
                    t += 0.016
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        def foreground_poller():
            while not stop_event.is_set():
                try:
                    _ = rec.poll_events()
                    _ = rec.get_continuous_gesture()
                    _ = rec.get_active_gesture_name()
                    time.sleep(0.002)
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=background_updater)
        t2 = threading.Thread(target=foreground_poller)
        t1.start()
        t2.start()

        time.sleep(0.15)
        stop_event.set()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors occurred: {errors}")

    def test_queue_preserved_on_temporary_hand_loss(self):
        """Verifies that transient 1-frame tracking loss does not discard queued events."""
        peace_hand = create_peace_hand()
        self.rec.update([peace_hand], now=1.0)
        self.rec.update([peace_hand], now=1.033)

        # Frame 3: tracking drops for 1 frame (motion blur or occlusion)
        self.rec.update([], now=1.066)

        # Event queue must NOT have been wiped!
        events = self.rec.poll_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, GestureType.PEACE)

    def test_fist_recognition_and_event_queue(self):
        """Verify closed fist is recognized, debounced over 2 frames, and emits FIST event."""
        fist_hand = create_fist_hand()
        self.rec.update([fist_hand], now=1.0)
        events = self.rec.poll_events()
        self.assertEqual(len(events), 0)

        self.rec.update([fist_hand], now=1.033)
        events = self.rec.poll_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, GestureType.FIST)
        self.assertEqual(self.rec.get_active_gesture_name(), GestureType.FIST)

    def test_opening_fist_does_not_trigger_snap(self):
        """Opening a closed fist to an open palm must NOT trigger a false snap event."""
        fist_hand = create_fist_hand()
        self.rec.update([fist_hand], now=1.0)
        self.rec.update([fist_hand], now=1.033)
        events = self.rec.poll_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, GestureType.FIST)

        # Open fist to open palm
        open_hand = create_open_palm_hand()
        self.rec.update([open_hand], now=1.066)
        self.rec.update([open_hand], now=1.100)
        open_events = self.rec.poll_events()
        for e in open_events:
            self.assertNotEqual(e.type, GestureType.SNAP)
        self.assertEqual(len(open_events), 1)
        self.assertEqual(open_events[0].type, GestureType.OPEN_PALM)

    def test_per_hand_gesture_isolation(self):
        """Verify that two hands can perform different gestures simultaneously without resetting each other."""
        rec = GestureRecognizer(cooldown=0.6)
        hand0 = create_peace_hand()
        hand1 = create_boost_hand()

        # Frame 1
        rec.update([hand0, hand1], now=1.0)
        events1 = rec.poll_events()
        self.assertEqual(len(events1), 0)

        # Frame 2: both hands confirmed
        rec.update([hand0, hand1], now=1.033)
        events2 = rec.poll_events()
        self.assertEqual(len(events2), 2)
        types = {e.type for e in events2}
        self.assertIn(GestureType.PEACE, types)
        self.assertIn(GestureType.BOOST, types)

    def test_boost_mode_responsiveness(self):
        """Verify boost mode provides faster trigger confirmation."""
        rec = GestureRecognizer(cooldown=0.6, boost_mode=True)
        hand = create_boost_hand()

        # Frame 1: in boost mode, triggers with 1 frame
        rec.update([hand], now=1.0)
        events = rec.poll_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, GestureType.BOOST)

    def test_multi_hand_snap_does_not_reset_other_hand(self):
        """Verify that Hand 0 snapping does not reset or suppress Hand 1's static gesture."""
        rec = GestureRecognizer(cooldown=0.6)
        hand0 = create_base_hand()
        scale0 = get_hand_scale(hand0)
        hand1 = create_peace_hand()

        # Frame 1: Hand 0 compressed, Hand 1 in Peace
        hand0[4] = make_lm(0.49, 0.43)
        hand0[12] = make_lm(0.50, 0.43)
        rec.update([hand0, hand1], now=1.0)
        events1 = rec.poll_events()
        self.assertEqual(len(events1), 0)

        # Frame 2: Hand 0 snaps, Hand 1 confirms Peace
        hand0[4] = make_lm(0.40, 0.40)
        hand0[12] = make_lm(0.55, 0.56)
        rec.update([hand0, hand1], now=1.033)
        events2 = rec.poll_events()
        self.assertEqual(len(events2), 2)
        event_types = {e.type for e in events2}
        self.assertIn(GestureType.SNAP, event_types)
        self.assertIn(GestureType.PEACE, event_types)
        snap_evt = next(e for e in events2 if e.type == GestureType.SNAP)
        peace_evt = next(e for e in events2 if e.type == GestureType.PEACE)
        self.assertEqual(snap_evt.hand_idx, 0)
        self.assertEqual(peace_evt.hand_idx, 1)

    def test_concurrent_two_hand_snaps(self):
        """Verify that both Hand 0 and Hand 1 snapping concurrently in the same frame are recognized."""
        rec = GestureRecognizer(cooldown=0.6)
        hand0 = create_base_hand()
        hand1 = create_base_hand()

        # Frame 1: Both hands compressed
        hand0[4] = make_lm(0.49, 0.43)
        hand0[12] = make_lm(0.50, 0.43)
        hand1[4] = make_lm(0.49, 0.43)
        hand1[12] = make_lm(0.50, 0.43)
        rec.update([hand0, hand1], now=1.0)
        self.assertEqual(len(rec.poll_events()), 0)

        # Frame 2: Both hands snap simultaneously
        hand0[4] = make_lm(0.40, 0.40)
        hand0[12] = make_lm(0.55, 0.56)
        hand1[4] = make_lm(0.40, 0.40)
        hand1[12] = make_lm(0.55, 0.56)
        rec.update([hand0, hand1], now=1.033)
        events = rec.poll_events()
        self.assertEqual(len(events), 2)
        hand_indices = {e.hand_idx for e in events}
        self.assertEqual(hand_indices, {0, 1})
        for e in events:
            self.assertEqual(e.type, GestureType.SNAP)

    def test_held_gesture_noise_immunity(self):
        """A single frame of classification dropout (noise) must NOT re-trigger a held gesture."""
        rec = GestureRecognizer(cooldown=0.4)
        hand = create_peace_hand()

        # Frame 1 & 2: trigger peace gesture
        rec.update([hand], now=1.0)
        rec.update([hand], now=1.033)
        events = rec.poll_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, GestureType.PEACE)

        # Intermediate noise frame (static_name None) at 1.5s (> cooldown)
        mock_noisy_hand = create_base_hand()  # intermediate, not peace
        rec.update([mock_noisy_hand], now=1.50)
        self.assertEqual(len(rec.poll_events()), 0)

        # Immediately back to peace on next frame (1.533s)
        rec.update([hand], now=1.533)
        events_after = rec.poll_events()
        self.assertEqual(len(events_after), 0, "Single noise frame must not cause duplicate re-trigger")

    def test_continuous_gesture_priority_in_active_gesture_name(self):
        """Active continuous gesture takes precedence in get_active_gesture_name()."""
        rec = GestureRecognizer(cooldown=0.6)
        t_up = create_thumbs_up_hand()

        rec.update([t_up], now=1.0)
        rec.update([t_up], now=1.033)
        self.assertEqual(rec.get_active_gesture_name(), GestureType.THUMBS_UP)


class TestTrackerAndAppGestureIntegration(unittest.TestCase):

    def test_hand_tracker_gesture_integration(self):
        """HandTracker processes landmarks and publishes active_gesture and events."""
        cam = CameraCapture(mock=True)
        tracker = HandTracker(cam)

        peace_hand = create_peace_hand()
        mock_res = SimpleNamespace(hand_landmarks=[peace_hand])

        # Frame 1 & 2
        tracker._process_landmarks(mock_res, 3.0)
        tracker._process_landmarks(mock_res, 3.0)

        state = tracker.get_state()
        self.assertEqual(state.active_gesture, GestureType.PEACE)

        events = tracker.poll_gesture_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, GestureType.PEACE)

    def test_app_gesture_event_handlers(self):
        """Verify VideoEffectsApp responds properly to all gesture events."""
        from main import VideoEffectsApp

        app = VideoEffectsApp(mock=True)

        shader_next_called = False
        def fake_next():
            nonlocal shader_next_called
            shader_next_called = True

        app.shader_manager = SimpleNamespace(
            next_shader=fake_next,
            active_shader_name="CyberGlitch",
            active_index=1,
            shader_count=10,
            last_error=None
        )

        hud_notifications = []
        box_visual = [True]
        def fake_notify(title, subtitle, color=None, duration=1.8):
            hud_notifications.append((title, subtitle))
        def fake_toggle_box():
            box_visual[0] = not box_visual[0]

        app.hud = SimpleNamespace(
            notify_gesture=fake_notify,
            toggle_box_visual=fake_toggle_box,
            show_box_visual=True,
            update_overlay=lambda *args, **kwargs: None
        )
        app.camera = SimpleNamespace(resolution=(640, 480), fps=30.0, backend_name="Mock")
        app.tracker = SimpleNamespace(get_state=lambda: TrackingData())

        app._handle_gesture_event(GestureEvent(type=GestureType.SNAP))
        self.assertTrue(shader_next_called)
        self.assertIn("SNAP", hud_notifications[-1][0])

        self.assertEqual(app.mask_mode, 0)
        app._handle_gesture_event(GestureEvent(type=GestureType.PEACE))
        self.assertEqual(app.mask_mode, 1)
        app._handle_gesture_event(GestureEvent(type=GestureType.PEACE))
        self.assertEqual(app.mask_mode, 2)
        app._handle_gesture_event(GestureEvent(type=GestureType.PEACE))
        self.assertEqual(app.mask_mode, 0)
        self.assertIn("PEACE", hud_notifications[-1][0])

        app._handle_gesture_event(GestureEvent(type=GestureType.FIST))
        self.assertFalse(box_visual[0])
        self.assertIn("FIST", hud_notifications[-1][0])

        app.intensity = 2.5
        app._handle_gesture_event(GestureEvent(type=GestureType.OPEN_PALM))
        self.assertEqual(app.intensity, 1.0)
        self.assertIn("OPEN PALM", hud_notifications[-1][0])

        app.intensity = 1.0
        app._handle_continuous_gesture(GestureType.THUMBS_UP, dt=0.5)
        self.assertAlmostEqual(app.intensity, 1.25, places=2)

        app._handle_continuous_gesture(GestureType.THUMBS_DOWN, dt=1.0)
        self.assertAlmostEqual(app.intensity, 0.75, places=2)

    def test_app_boost_gesture_handler(self):
        """Verify BOOST gesture event toggles boost mode in VideoEffectsApp."""
        from main import VideoEffectsApp

        app = VideoEffectsApp(mock=True)
        self.assertFalse(app.boost_mode)

        app._handle_gesture_event(GestureEvent(type=GestureType.BOOST))
        self.assertTrue(app.boost_mode)

        app._handle_gesture_event(GestureEvent(type=GestureType.BOOST))
        self.assertFalse(app.boost_mode)

    def test_thumbs_up_rejection_fist_and_framing(self):
        """Verify check_thumbs_up rejects natural fists and 1-hand framing gestures."""
        # 1. Natural fist with slightly raised thumb tip
        fist = create_fist_hand()
        fist[4] = make_lm(0.48, 0.518)
        scale = get_hand_scale(fist)
        self.assertFalse(check_thumbs_up(fist, scale))
        self.assertTrue(check_fist(fist, scale))

        # 2. 1-hand framing hand: thumb up, index extended horizontally
        frame_hand = create_fist_hand()
        frame_scale = get_hand_scale(frame_hand)
        frame_hand[2] = make_lm(0.45, 0.60)
        frame_hand[3] = make_lm(0.45, 0.50)
        frame_hand[4] = make_lm(0.45, 0.36)  # thumb up
        frame_hand[6] = make_lm(0.48, 0.45)
        frame_hand[7] = make_lm(0.54, 0.45)
        frame_hand[8] = make_lm(0.60, 0.45)  # index extended horizontally
        self.assertFalse(check_thumbs_up(frame_hand, frame_scale))

    def test_two_hands_present_does_not_trigger_continuous_thumbs(self):
        """When 2 hands are present (e.g. framing a box), continuous thumbs up/down must not trigger."""
        rec = GestureRecognizer(cooldown=0.6)
        t_up = create_thumbs_up_hand()
        fist = create_fist_hand()

        # Both hands in view (e.g. framing or multi-hand interaction)
        rec.update([t_up, fist], now=1.0)
        rec.update([t_up, fist], now=1.033)
        self.assertIsNone(rec.get_continuous_gesture(), "Continuous thumbs must be suppressed when 2 hands are present")

    def test_thumbs_up_rejection_pinch_and_snap_prep(self):
        """Verify check_thumbs_up rejects 1-hand pinch and snap prep configurations."""
        # 1-hand pinch: thumb tip touches index tip near (0.45, 0.49)
        pinch_hand = create_fist_hand()
        pinch_hand[2] = make_lm(0.45, 0.60)
        pinch_hand[3] = make_lm(0.45, 0.53)
        pinch_hand[4] = make_lm(0.45, 0.49)
        pinch_hand[8] = make_lm(0.45, 0.50)
        scale = get_hand_scale(pinch_hand)
        self.assertFalse(check_thumbs_up(pinch_hand, scale), "Pinch must not trigger thumbs up")

        # Snap prep: thumb tip touches middle tip near (0.48, 0.48)
        snap_hand = create_fist_hand()
        snap_hand[2] = make_lm(0.45, 0.60)
        snap_hand[3] = make_lm(0.45, 0.53)
        snap_hand[4] = make_lm(0.48, 0.48)
        snap_hand[12] = make_lm(0.49, 0.48)
        snap_scale = get_hand_scale(snap_hand)
        self.assertFalse(check_thumbs_up(snap_hand, snap_scale), "Snap prep must not trigger thumbs up")

    def test_thumbs_up_rejection_pointing_and_extended_fingers(self):
        """Verify check_thumbs_up rejects hands with pointing index or extended ring/pinky."""
        # Pointing index finger up
        point_hand = create_fist_hand()
        point_hand[6] = make_lm(0.45, 0.40)
        point_hand[8] = make_lm(0.45, 0.25)  # index pointing straight up
        point_hand[2] = make_lm(0.43, 0.60)
        point_hand[3] = make_lm(0.43, 0.50)
        point_hand[4] = make_lm(0.43, 0.36)  # thumb also up
        scale = get_hand_scale(point_hand)
        self.assertFalse(check_thumbs_up(point_hand, scale), "Pointing finger must not trigger thumbs up")

        # Extended pinky (drinking tea pose with thumb up)
        tea_hand = create_thumbs_up_hand()
        tea_hand[18] = make_lm(0.58, 0.45)
        tea_hand[20] = make_lm(0.58, 0.32)  # pinky extended
        scale_tea = get_hand_scale(tea_hand)
        self.assertFalse(check_thumbs_up(tea_hand, scale_tea), "Extended pinky must not trigger thumbs up")

    def test_thumbs_down_rejection_pinch_and_snap_prep(self):
        """Verify check_thumbs_down rejects pinch and snap prep."""
        pinch_hand = create_fist_hand()
        pinch_hand[4] = make_lm(0.45, 0.55)
        pinch_hand[8] = make_lm(0.45, 0.55)
        scale = get_hand_scale(pinch_hand)
        self.assertFalse(check_thumbs_down(pinch_hand, scale), "Pinch must not trigger thumbs down")

    def test_continuous_thumbs_requires_multiple_frames_in_boost_mode(self):
        """Even in boost mode, a single noise frame must not trigger continuous thumbs up."""
        rec = GestureRecognizer(cooldown=0.6, boost_mode=True)
        t_up = create_thumbs_up_hand()

        # Single frame
        rec.update([t_up], now=1.0)
        self.assertIsNone(rec.get_continuous_gesture(), "Single frame should not activate continuous thumbs up")

        # Second consecutive frame
        rec.update([t_up], now=1.033)
        self.assertEqual(rec.get_continuous_gesture(), GestureType.THUMBS_UP, "Two frames confirm continuous thumbs up")

    def test_shape_mode_suppresses_static_and_continuous_gestures(self):
        """When shape_mode is True, static gestures and continuous gestures are strictly suppressed."""
        rec = GestureRecognizer(cooldown=0.6)
        fist = create_fist_hand()
        peace = create_peace_hand()
        t_up = create_thumbs_up_hand()

        # Fist with shape_mode=True
        rec.update([fist], now=1.0, shape_mode=True)
        events = rec.poll_events()
        self.assertEqual(len(events), 0, "Shape mode must suppress fist events")

        # Peace with shape_mode=True
        rec.update([peace], now=2.0, shape_mode=True)
        events = rec.poll_events()
        self.assertEqual(len(events), 0, "Shape mode must suppress peace events")

        # Thumbs up with shape_mode=True
        rec.update([t_up], now=3.0, shape_mode=True)
        rec.update([t_up], now=3.033, shape_mode=True)
        self.assertIsNone(rec.get_continuous_gesture(), "Shape mode must suppress continuous thumbs up")

    def test_pinch_rejected_from_fist_detection(self):
        """A pinch pose (thumb and index fingertips touching) must never be recognized as a fist."""
        pinch_hand = create_fist_hand()
        # Separate thumb and index slightly from palm and bring tips together
        pinch_hand[4] = make_lm(0.46, 0.45)  # Thumb tip
        pinch_hand[8] = make_lm(0.47, 0.45)  # Index tip
        scale = get_hand_scale(pinch_hand)

        self.assertFalse(check_fist(pinch_hand, scale), "Pinch must be rejected from fist detection")


if __name__ == "__main__":
    unittest.main()


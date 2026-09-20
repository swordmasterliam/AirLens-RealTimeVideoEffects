import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


class GestureType:
    NONE = "none"
    SNAP = "snap"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    BOOST = "boost"


@dataclass
class GestureEvent:
    type: str
    hand_idx: int = 0
    timestamp: float = 0.0
    details: str = ""


def point_dist(p1, p2) -> float:
    dx = float(p1.x - p2.x)
    dy = float(p1.y - p2.y)
    return (dx * dx + dy * dy) ** 0.5


def get_hand_scale(hand) -> float:
    """Calculate hand scale from wrist (being 0) to middle MCP (being 9) distance."""
    if len(hand) < 21:
        return 0.2
    scale = point_dist(hand[0], hand[9])
    return max(scale, 0.05)


def is_finger_extended(hand, tip_idx: int, pip_idx: int, mcp_idx: int, scale: float) -> bool:
    d_tip_wrist = point_dist(hand[tip_idx], hand[0])
    d_pip_wrist = point_dist(hand[pip_idx], hand[0])
    d_tip_mcp = point_dist(hand[tip_idx], hand[mcp_idx])
    return (d_tip_wrist > d_pip_wrist) and (d_tip_mcp > 0.85 * scale)


def is_finger_folded(hand, tip_idx: int, pip_idx: int, mcp_idx: int, scale: float) -> bool:
    d_tip_wrist = point_dist(hand[tip_idx], hand[0])
    d_pip_wrist = point_dist(hand[pip_idx], hand[0])
    d_tip_mcp = point_dist(hand[tip_idx], hand[mcp_idx])
    return (d_tip_mcp < 0.58 * scale) or ((d_tip_wrist < d_pip_wrist * 0.98) and (d_tip_mcp < 0.72 * scale))


def is_thumb_extended(hand, scale: float) -> bool:
    d_tip_mcp = point_dist(hand[4], hand[2])
    d_tip_middle_mcp = point_dist(hand[4], hand[9])
    return (d_tip_mcp > 0.45 * scale) and (d_tip_middle_mcp > 0.65 * scale)


def is_thumb_folded(hand, scale: float) -> bool:
    d_tip_middle_mcp = point_dist(hand[4], hand[9])
    d_tip_index_mcp = point_dist(hand[4], hand[5])
    return (d_tip_middle_mcp < 0.65 * scale) or (d_tip_index_mcp < 0.55 * scale)


def check_peace(hand, scale: float) -> bool:
    if len(hand) < 21:
        return False

    if not (is_finger_extended(hand, 8, 6, 5, scale) and
            is_finger_extended(hand, 12, 10, 9, scale) and
            is_finger_folded(hand, 16, 14, 13, scale) and
            is_finger_folded(hand, 20, 18, 17, scale)):
        return False

    if is_thumb_extended(hand, scale):
        return False

    d_v = point_dist(hand[8], hand[12])
    return d_v > 0.15 * scale


def check_thumbs_up(hand, scale: float) -> bool:
    if len(hand) < 21:
        return False

    if not (is_finger_folded(hand, 8, 6, 5, scale) and
            is_finger_folded(hand, 12, 10, 9, scale) and
            is_finger_folded(hand, 16, 14, 13, scale) and
            is_finger_folded(hand, 20, 18, 17, scale)):
        return False

    #guardin against 1-hand framing, pointing, or like partially extended fingers
    if (is_finger_extended(hand, 8, 6, 5, scale) or
        is_finger_extended(hand, 12, 10, 9, scale) or
        is_finger_extended(hand, 16, 14, 13, scale) or
        is_finger_extended(hand, 20, 18, 17, scale)):
        return False
    if (point_dist(hand[8], hand[5]) > 0.48 * scale or
        point_dist(hand[12], hand[9]) > 0.48 * scale or
        point_dist(hand[16], hand[13]) > 0.48 * scale or
        point_dist(hand[20], hand[17]) > 0.48 * scale):
        return False
    if abs(float(hand[8].x - hand[5].x)) > 0.35 * scale:
        return False

    # now gaurding against 1-hand pinch (thumb touching index) or snap prep (thumb touching middle) but this isnt working perfectly for the pinky in the snap gesture
    if  point_dist(hand[4], hand[8]) < 0.38 * scale:
        return False
    if point_dist(hand[4], hand[12]) < 0.35 * scale:
        return False

    d_thumb = point_dist(hand[4], hand[2])
    if d_thumb < 0.35 * scale:
        return False

    dy = float(hand[4].y - hand[2].y)
    dx = float(hand[4].x - hand[2].x)
    dy_norm = dy / scale

    # thumb segment from IP (3) to TIP (4)
    dy_ip = float(hand[4].y - hand[3].y)
    dx_ip = float(hand[4].x - hand[3].x)

    if (dy_norm < -0.28 and
        abs(dy) > abs(dx) * 0.9 and
        dy_ip < -0.10 * scale and
        abs(dy_ip) > abs(dx_ip) * 0.8 and
        hand[4].y < hand[3].y and
        hand[4].y < hand[2].y and
        hand[4].y < hand[5].y and
        hand[4].y < hand[8].y and
        hand[4].y < hand[12].y and
        hand[4].y < hand[16].y and
        hand[4].y < hand[20].y and
        hand[4].y < hand[0].y - 0.20 * scale and
        hand[0].y > hand[9].y):
        return True
    return False


def check_thumbs_down(hand, scale: float) -> bool:
    if len(hand) < 21:
        return False

    if not (is_finger_folded(hand, 8, 6, 5, scale) and
            is_finger_folded(hand, 12, 10, 9, scale) and
            is_finger_folded(hand, 16, 14, 13, scale) and
            is_finger_folded(hand, 20, 18, 17, scale)):
        return False

    # more guard against 1-hand framing, pointing, or partially extended fingers
    if (is_finger_extended(hand, 8, 6, 5, scale) or
        is_finger_extended(hand, 12, 10, 9, scale) or
        is_finger_extended(hand, 16, 14, 13, scale) or
        is_finger_extended(hand, 20, 18, 17, scale)):
        return False
    if (point_dist(hand[8], hand[5]) > 0.48 * scale or
        point_dist(hand[12], hand[9]) > 0.48 * scale or
        point_dist(hand[16], hand[13]) > 0.48 * scale or
        point_dist(hand[20], hand[17]) > 0.48 * scale):
        return False
    if abs(float(hand[8].x - hand[5].x)) > 0.35 * scale:
        return False

    # guard against 1-hand pinch or snap prep
    if point_dist(hand[4], hand[8]) < 0.38 * scale:
        return False
    if point_dist(hand[4], hand[12]) < 0.35 * scale:
        return False

    d_thumb = point_dist(hand[4], hand[2])
    if d_thumb < 0.35 * scale:
        return False

    dy = float(hand[4].y - hand[2].y)
    dx = float(hand[4].x - hand[2].x)
    dy_norm = dy / scale

    dy_ip = float(hand[4].y - hand[3].y)
    dx_ip = float(hand[4].x - hand[3].x)

    if (dy_norm > 0.22 and
        abs(dy) > abs(dx) * 0.85 and
        dy_ip > 0.08 * scale and
        abs(dy_ip) > abs(dx_ip) * 0.7 and
        hand[4].y > hand[3].y and
        hand[4].y > hand[2].y and
        hand[4].y > hand[8].y and
        hand[4].y > hand[12].y and
        hand[4].y > hand[16].y and
        hand[4].y > hand[20].y):
        return True
    return False


def check_fist(hand, scale: float) -> bool:
    if len(hand) < 21:
        return False

    if not (is_finger_folded(hand, 8, 6, 5, scale) and
            is_finger_folded(hand, 12, 10, 9, scale) and
            is_finger_folded(hand, 16, 14, 13, scale) and
            is_finger_folded(hand, 20, 18, 17, scale)):
        return False

    if check_thumbs_up(hand, scale) or check_thumbs_down(hand, scale):
        return False

    # Strict isolation: if thumb is extended outward or pointing up/down/sideways, reject the fist
    if is_thumb_extended(hand, scale):
        return False

    # Avoid classifying pinch (thumb touching index tip) as a fist 🙏
    d_thumb_index = point_dist(hand[4], hand[8]) / scale
    if d_thumb_index < 0.38:
        if point_dist(hand[8], hand[0]) >= point_dist(hand[6], hand[0]) * 0.95 or hand[4].y < hand[5].y - 0.15 * scale:
            return False

    # Avoid classifying snap prep as a fist please!
    d_thumb_middle = point_dist(hand[4], hand[12]) / scale
    if d_thumb_middle < 0.38:
        if point_dist(hand[12], hand[0]) >= point_dist(hand[10], hand[0]) * 0.95 or hand[4].y < hand[9].y - 0.15 * scale:
            return False

    # All fingers MUST be deeply tucked (ts is never going to work)
    if (point_dist(hand[8], hand[5]) > 0.48 * scale or
        point_dist(hand[12], hand[9]) > 0.48 * scale or
        point_dist(hand[16], hand[13]) > 0.48 * scale or
        point_dist(hand[20], hand[17]) > 0.48 * scale):
        return False

    return is_thumb_folded(hand, scale) or point_dist(hand[4], hand[9]) < 0.65 * scale


def check_open_palm(hand, scale: float) -> bool:
    if len(hand) < 21:
        return False

    return (is_finger_extended(hand, 8, 6, 5, scale) and
            is_finger_extended(hand, 12, 10, 9, scale) and
            is_finger_extended(hand, 16, 14, 13, scale) and
            is_finger_extended(hand, 20, 18, 17, scale) and
            is_thumb_extended(hand, scale))


def check_boost(hand, scale: float) -> bool:
    if len(hand) < 21:
        return False

    if not (is_finger_extended(hand, 8, 6, 5, scale) and
            is_finger_extended(hand, 20, 18, 17, scale) and
            is_finger_folded(hand, 12, 10, 9, scale) and
            is_finger_folded(hand, 16, 14, 13, scale)):
        return False

    d_v = point_dist(hand[8], hand[20])
    return d_v > 0.25 * scale


class SnapDetector:
    def __init__(self, cooldown: float = 0.65, boost_mode: bool = False):
        self.cooldown = cooldown
        self.boost_mode = boost_mode
        self.last_snap_time = -10.0

        self.state = "IDLE"
        self.last_compress_time = 0.0
        self.min_dist_middle = 1.0
        self.min_dist_index = 1.0
        self.prev_dist_middle = 1.0
        self.prev_dist_index = 1.0
        self.prev_time = 0.0

    def set_boost(self, enabled: bool):
        self.boost_mode = enabled

    def reset(self):
        self.state = "IDLE"
        self.last_compress_time = 0.0
        self.min_dist_middle = 1.0
        self.min_dist_index = 1.0
        self.prev_dist_middle = 1.0
        self.prev_dist_index = 1.0
        self.prev_time = 0.0

    @property
    def is_compressed(self) -> bool:
        return self.state == "COMPRESSED"

    def update(self, hand, scale: float, now: float) -> bool:
        if len(hand) < 21:
            return False

        effective_cooldown = max(0.40, self.cooldown * 0.75) if self.boost_mode else self.cooldown
        if (now - self.last_snap_time) < effective_cooldown:
            return False

        if check_boost(hand, scale):
            if self.state == "COMPRESSED":
                self.reset()
            return False

        if self.state == "IDLE" and check_fist(hand, scale):
            return False

        d_middle = point_dist(hand[4], hand[12]) / scale
        d_index = point_dist(hand[4], hand[8]) / scale

        dt = now - self.prev_time if self.prev_time > 0.0 else 0.033
        self.prev_time = now

        v_middle = (d_middle - self.prev_dist_middle) / max(dt, 0.005)
        v_index = (d_index - self.prev_dist_index) / max(dt, 0.005)

        self.prev_dist_middle = d_middle
        self.prev_dist_index = d_index

        PINCH_THRESH = 0.38 if self.boost_mode else 0.35
        RELEASE_THRESH = 0.46 if self.boost_mode else 0.50
        MIN_JUMP = 0.22 if self.boost_mode else 0.26
        MAX_RELEASE_WINDOW = 0.20 if self.boost_mode else 0.16
        V_THRESH = 1.5 if self.boost_mode else 1.8

        is_pinching_middle = (d_middle <= PINCH_THRESH)
        is_pinching_index = (d_index <= PINCH_THRESH)

        if is_pinching_middle or is_pinching_index:
            self.state = "COMPRESSED"
            self.last_compress_time = now
            if is_pinching_middle:
                self.min_dist_middle = min(self.min_dist_middle, d_middle)
            if is_pinching_index:
                self.min_dist_index = min(self.min_dist_index, d_index)
            return False

        elif self.state == "COMPRESSED":
            elapsed_since_compress = now - self.last_compress_time
            jump_middle = d_middle - self.min_dist_middle
            jump_index = d_index - self.min_dist_index

            snap_middle = (
                jump_middle >= MIN_JUMP and
                elapsed_since_compress <= MAX_RELEASE_WINDOW and
                (v_middle > V_THRESH or d_middle >= RELEASE_THRESH)
            )

            snap_index = (
                jump_index >= MIN_JUMP and
                elapsed_since_compress <= MAX_RELEASE_WINDOW and
                (v_index > V_THRESH or d_index >= RELEASE_THRESH)
            )

            if snap_middle or snap_index:
                self.last_snap_time = now
                self.reset()
                return True

            if check_fist(hand, scale) or elapsed_since_compress > MAX_RELEASE_WINDOW or (d_middle > 0.70 and d_index > 0.70):
                self.reset()

        return False


class GestureRecognizer:
    def __init__(self, cooldown: float = 0.65, boost_mode: bool = False):
        self.cooldown = cooldown
        self.boost_mode = boost_mode
        self.snap_detectors = [
            SnapDetector(cooldown=cooldown, boost_mode=boost_mode) for _ in range(4)
        ]
        self._lock = threading.RLock()
        self._event_queue: List[GestureEvent] = []
        self._active_continuous: Optional[str] = None
        self._active_gesture_name: str = GestureType.NONE
        self._post_snap_suppress_until: List[float] = [0.0] * 4

        self._per_hand_static_state: List[Dict[str, Dict[str, Any]]] = []
        for _ in range(4):
            hand_state = {}
            for g in (GestureType.PEACE, GestureType.FIST, GestureType.OPEN_PALM, GestureType.BOOST):
                hand_state[g] = {
                    "count": 0,
                    "release_count": 0,
                    "last_trigger": -10.0,
                    "held": False
                }
            self._per_hand_static_state.append(hand_state)

        self._static_state = self._per_hand_static_state[0]

        self._continuous_candidate: Optional[str] = None
        self._continuous_count: int = 0

    def set_boost(self, enabled: bool):
        with self._lock:
            self.boost_mode = enabled
            for sd in list(self.snap_detectors):
                sd.set_boost(enabled)

    def _get_hand_static_state(self, hand_idx: int) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            while len(self._per_hand_static_state) <= hand_idx:
                hand_state = {}
                for g in (GestureType.PEACE, GestureType.FIST, GestureType.OPEN_PALM, GestureType.BOOST):
                    hand_state[g] = {
                        "count": 0,
                        "release_count": 0,
                        "last_trigger": -10.0,
                        "held": False
                    }
                self._per_hand_static_state.append(hand_state)
            return self._per_hand_static_state[hand_idx]

    def reset(self):
        with self._lock:
            self._active_continuous = None
            self._active_gesture_name = GestureType.NONE
            for sd in list(self.snap_detectors):
                sd.reset()
            for h_state in list(self._per_hand_static_state):
                for g in h_state:
                    h_state[g]["count"] = 0
                    h_state[g]["release_count"] = 0
                    h_state[g]["held"] = False
            self._continuous_candidate = None
            self._continuous_count = 0
            self._post_snap_suppress_until = [0.0] * max(4, len(self.snap_detectors))

    def clear_events(self):
        with self._lock:
            self._event_queue.clear()

    def update(self, landmarks_list, now: Optional[float] = None, shape_mode: bool = False):
        if now is None:
            now = time.perf_counter()

        if not landmarks_list:
            self.reset()
            return

        current_active = GestureType.NONE
        detected_snap_hands = set()
        detected_continuous = None
        detected_static_per_hand: List[Optional[str]] = []

        for hand_idx, hand in enumerate(landmarks_list[:2]):
            if len(hand) < 21:
                detected_static_per_hand.append(None)
                continue

            scale = get_hand_scale(hand)

            with self._lock:
                while len(self.snap_detectors) <= hand_idx:
                    self.snap_detectors.append(SnapDetector(cooldown=self.cooldown, boost_mode=self.boost_mode))
                while len(self._post_snap_suppress_until) <= hand_idx:
                    self._post_snap_suppress_until.append(0.0)
                sd = self.snap_detectors[hand_idx]

            if sd.update(hand, scale, now):
                detected_snap_hands.add(hand_idx)
                with self._lock:
                    while len(self._post_snap_suppress_until) <= hand_idx:
                        self._post_snap_suppress_until.append(0.0)
                    self._post_snap_suppress_until[hand_idx] = now + 0.40
                    self._event_queue.append(GestureEvent(
                        type=GestureType.SNAP,
                        hand_idx=hand_idx,
                        timestamp=now,
                        details="Finger Snap Detected"
                    ))
                current_active = GestureType.SNAP
                detected_static_per_hand.append(None)
                continue

            if not shape_mode and detected_continuous is None and not sd.is_compressed and len(landmarks_list) == 1:
                if check_thumbs_up(hand, scale):
                    detected_continuous = GestureType.THUMBS_UP
                elif check_thumbs_down(hand, scale):
                    detected_continuous = GestureType.THUMBS_DOWN

            # isolated per-hand static check
            cand_static = None
            is_boost = check_boost(hand, scale)
            with self._lock:
                suppress_until = self._post_snap_suppress_until[hand_idx] if hand_idx < len(self._post_snap_suppress_until) else 0.0
            if (not sd.is_compressed or is_boost) and (now >= suppress_until):
                if is_boost:
                    cand_static = GestureType.BOOST
                elif not shape_mode:
                    if check_peace(hand, scale):
                        cand_static = GestureType.PEACE
                    elif check_fist(hand, scale):
                        cand_static = GestureType.FIST
                    elif check_open_palm(hand, scale):
                        cand_static = GestureType.OPEN_PALM

            detected_static_per_hand.append(cand_static)

        # Update perhand static states with isolation
        effective_cooldown = max(0.40, self.cooldown * 0.75) if self.boost_mode else self.cooldown
        count_req = 1 if (self.boost_mode and detected_continuous is None) else 2

        with self._lock:
            for h_idx, static_name in enumerate(detected_static_per_hand):
                h_state = self._get_hand_static_state(h_idx)
                if h_idx in detected_snap_hands:
                    for g_name, state in h_state.items():
                        state["count"] = 0
                        state["release_count"] = 0
                        state["held"] = False
                    continue

                for g_name, state in h_state.items():
                    if static_name == g_name:
                        state["release_count"] = 0
                        state["count"] += 1
                        if current_active == GestureType.NONE:
                            current_active = g_name
                        if state["count"] >= count_req and not state["held"] and (now - state["last_trigger"]) >= effective_cooldown:
                            state["held"] = True
                            state["last_trigger"] = now
                            self._event_queue.append(GestureEvent(
                                type=g_name,
                                hand_idx=h_idx,
                                timestamp=now,
                                details=f"{g_name.replace('_', ' ').title()} Gesture"
                            ))
                    else:
                        state["count"] = 0
                        if static_name is not None:
                            state["held"] = False
                            state["release_count"] = 0
                        else: #needs 2+ consecutive frames
                            state["release_count"] = state.get("release_count", 0) + 1
                            if state["release_count"] >= 2:
                                state["held"] = False

            for h_idx in range(len(detected_static_per_hand), len(self._per_hand_static_state)):
                h_state = self._per_hand_static_state[h_idx]
                for g_name, state in h_state.items():
                    state["count"] = 0
                    state["release_count"] = state.get("release_count", 0) + 1
                    if state["release_count"] >= 2:
                        state["held"] = False

            if detected_continuous is not None and not detected_snap_hands:
                if current_active == GestureType.NONE:
                    current_active = detected_continuous
                if self._continuous_candidate == detected_continuous:
                    self._continuous_count += 1
                else:
                    self._continuous_candidate = detected_continuous
                    self._continuous_count = 1

                cont_thresh = 2
                if self._continuous_count >= cont_thresh:
                    self._active_continuous = detected_continuous
            else:
                self._continuous_candidate = None
                self._continuous_count = 0
                self._active_continuous = None

            self._active_gesture_name = current_active

    def poll_events(self) -> List[GestureEvent]:
        with self._lock:
            if not self._event_queue:
                return []
            events = list(self._event_queue)
            self._event_queue.clear()
            return events

    def get_continuous_gesture(self) -> Optional[str]:
        with self._lock:
            return self._active_continuous

    def get_active_gesture_name(self) -> str:
        with self._lock:
            if self._active_continuous:
                return self._active_continuous
            return self._active_gesture_name

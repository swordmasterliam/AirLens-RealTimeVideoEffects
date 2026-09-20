import os
import time
import threading
import urllib.request
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"


from gesture import GestureRecognizer, GestureType, point_dist
from face_tracker import FaceTracker


@dataclass
class TrackingData:
    has_box: float = 0.0                                            # 1.0 if box active, 0.0 if not
    box: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)  # (min_x, min_y, max_x, max_y)
    hands_count: int = 0                                            # Number of hands visible (0, 1, 2)
    thumb1: Tuple[float, float] = (0.0, 0.0)                         # Normalized (x, y)
    index1: Tuple[float, float] = (0.0, 0.0)
    thumb2: Tuple[float, float] = (0.0, 0.0)
    index2: Tuple[float, float] = (0.0, 0.0)
    inference_time_ms: float = 0.0
    tracking_fps: float = 0.0
    active_gesture: str = "none"
    boost_active: bool = False
    wink_detected: bool = False


class AdaptiveSmoother:
    def __init__(self, min_alpha=0.35, max_alpha=0.85, velocity_scale=5.0, boost_mode=False):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.velocity_scale = velocity_scale
        self.boost_mode = boost_mode
        self.prev_val = None

    def set_boost(self, enabled: bool):
        self.boost_mode = enabled

    def update(self, new_val: np.ndarray) -> np.ndarray:
        new_val = np.asarray(new_val, dtype=np.float32)
        if not np.all(np.isfinite(new_val)):
            if self.prev_val is not None:
                return self.prev_val.copy()
            return np.zeros_like(new_val)

        if self.prev_val is None:
            self.prev_val = new_val.copy()
            return new_val.copy()

        diff = np.linalg.norm(new_val - self.prev_val)
        v_scale = self.velocity_scale * 1.5 if self.boost_mode else self.velocity_scale
        max_a = min(0.98, self.max_alpha + 0.08) if self.boost_mode else self.max_alpha
        min_a = max(0.20, self.min_alpha - 0.05) if self.boost_mode else self.min_alpha

        alpha = min_a + (max_a - min_a) * np.clip(diff * v_scale, 0.0, 1.0)
        smoothed = alpha * new_val + (1.0 - alpha) * self.prev_val
        self.prev_val = smoothed
        return smoothed

    def reset(self):
        self.prev_val = None


class HandTracker:
    def __init__(self, camera_capture, model_path=None, boost: bool = False):
        self.camera = camera_capture
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        self._state = TrackingData()

        self.boost_mode = boost
        self._prev_hand_centers = []

        self._box_smoother = AdaptiveSmoother(min_alpha=0.30, max_alpha=0.90, velocity_scale=6.0, boost_mode=boost)
        self._f1_smoother = AdaptiveSmoother(min_alpha=0.35, max_alpha=0.90, velocity_scale=6.0, boost_mode=boost)
        self._f2_smoother = AdaptiveSmoother(min_alpha=0.35, max_alpha=0.90, velocity_scale=6.0, boost_mode=boost)

        self._last_frame_id = -1
        self._track_count = 0
        self._track_fps = 0.0
        self._fps_timer = time.perf_counter()

        self.extra_gestures_enabled = True
        self.gesture_recognizer = GestureRecognizer(cooldown=0.65, boost_mode=boost)
        self.face_tracker = FaceTracker()
        self._wink_event = False

    def set_boost(self, enabled: bool):
        with self._lock:
            self.boost_mode = enabled
            self._box_smoother.set_boost(enabled)
            self._f1_smoother.set_boost(enabled)
            self._f2_smoother.set_boost(enabled)
            self.gesture_recognizer.set_boost(enabled)

    def set_extra_gestures_enabled(self, enabled: bool):
        with self._lock:
            self.extra_gestures_enabled = enabled
            if not enabled:
                self.gesture_recognizer.reset()
                self.gesture_recognizer.clear_events()

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"[Tracker] Model not found at {self.model_path}. Downloading from google...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print(f"[Tracker] Download complete ({os.path.getsize(self.model_path)} bytes).")
            except Exception as e:
                print(f"[Tracker] Failed to download model: {e}")
                return False
        return True

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._tracking_worker, daemon=True, name="HandTrackerWorker")
        self._thread.start()

    def _init_detector(self):
        if not MEDIAPIPE_AVAILABLE:
            print("[Tracker] MediaPipe library not available Tracking disabled.")
            return None

        if not self._ensure_model_exists():
            return None

        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            detector = mp_vision.HandLandmarker.create_from_options(options)
            print("[Tracker] MediaPipe HandLandmarker initialized successfully.")
            return detector
        except Exception as e:
            print(f"[Tracker] Error initializing HandLandmarker: {e}")
            return None

    def _tracking_worker(self):
        detector = self._init_detector()

        while self._running:
            frame, frame_id, _ = self.camera.get_latest_frame()
            if frame is None or frame_id == self._last_frame_id:
                time.sleep(0.002)
                continue

            self._last_frame_id = frame_id

            if detector is None:
                if self.camera and getattr(self.camera, "mock", False):
                    self._process_mock_gesture()
                time.sleep(0.016)
                continue

            t0 = time.perf_counter()

            h, w = frame.shape[:2]
            target_w = 480 if self.boost_mode else 320
            target_h = int(h * (target_w / w))
            small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

            try:
                result = detector.detect(mp_image)
            except Exception as e:
                time.sleep(0.005)
                continue

            t_infer = (time.perf_counter() - t0) * 1000.0

            if self.face_tracker:
                try:
                    if self.face_tracker.process_frame(rgb_small, now=time.perf_counter()):
                        with self._lock:
                            self._wink_event = True
                except Exception:
                    pass

            try:
                self._process_landmarks(result, t_infer)
            except Exception as e:
                print(f"[Tracker] Error processing landmarks: {e}")
                time.sleep(0.005)
                continue

            self._track_count += 1
            elapsed = time.perf_counter() - self._fps_timer
            if elapsed >= 1.0:
                self._track_fps = self._track_count / elapsed
                self._track_count = 0
                self._fps_timer = time.perf_counter()

        if detector:
            detector.close()
        print("[Tracker] Stopped.")

    def _process_mock_gesture(self):
        t = time.perf_counter()
        cx = 0.5 + 0.12 * np.sin(t * 0.9)
        cy = 0.5 + 0.08 * np.cos(t * 0.7)
        bw = 0.32 + 0.08 * np.sin(t * 1.3)
        bh = 0.24 + 0.06 * np.cos(t * 1.1)

        min_x = float(np.clip(cx - bw * 0.5, 0.05, 0.95))
        max_x = float(np.clip(cx + bw * 0.5, 0.05, 0.95))
        min_y = float(np.clip(cy - bh * 0.5, 0.05, 0.95))
        max_y = float(np.clip(cy + bh * 0.5, 0.05, 0.95))

        t1 = (min_x, max_y)
        i1 = (min_x, min_y)
        t2 = (max_x, min_y)
        i2 = (max_x, max_y)

        with self._lock:
            self._state = TrackingData(
                has_box=1.0,
                box=(min_x, min_y, max_x, max_y),
                hands_count=2,
                thumb1=t1,
                index1=i1,
                thumb2=t2,
                index2=i2,
                inference_time_ms=0.5,
                tracking_fps=60.0,
                active_gesture="box_only" if not self.extra_gestures_enabled else self.gesture_recognizer.get_active_gesture_name(),
                boost_active=self.boost_mode
            )

    def _process_landmarks(self, result, inference_ms: float):
        raw_landmarks = (result.hand_landmarks if (result and getattr(result, "hand_landmarks", None) is not None) else [])
        landmarks_list = [h for h in raw_landmarks if h is not None and len(h) >= 21]
        num_hands = len(landmarks_list)
        now = time.perf_counter()

        def get_hand_pos(hand):
            if hand is None or len(hand) < 21:
                return (0.0, 0.0)
            return (float(hand[4].x + hand[8].x) * 0.5, float(hand[4].y + hand[8].y) * 0.5)

        if num_hands == 0:
            self._prev_hand_centers = []
            if self.extra_gestures_enabled:
                self.gesture_recognizer.update([], now=now)

            if self.camera and getattr(self.camera, "mock", False):
                self._process_mock_gesture()
                return

            with self._lock:
                prev = self._state
                new_has_box = max(0.0, prev.has_box - 0.15)
                self._state = TrackingData(
                    has_box=new_has_box,
                    box=prev.box,
                    hands_count=0,
                    thumb1=prev.thumb1,
                    index1=prev.index1,
                    thumb2=prev.thumb2,
                    index2=prev.index2,
                    inference_time_ms=inference_ms,
                    tracking_fps=self._track_fps,
                    active_gesture="none",
                    boost_active=self.boost_mode
                )
            self._box_smoother.reset()
            self._f1_smoother.reset()
            self._f2_smoother.reset()
            return

        # Stable hand tracking matching
        if num_hands >= 2:
            h_curr = landmarks_list[:2]
            if len(self._prev_hand_centers) == 2:
                p0, p1 = self._prev_hand_centers
                c0 = get_hand_pos(h_curr[0])
                c1 = get_hand_pos(h_curr[1])

                cost_direct = ((c0[0]-p0[0])**2 + (c0[1]-p0[1])**2) + ((c1[0]-p1[0])**2 + (c1[1]-p1[1])**2)
                cost_swapped = ((c1[0]-p0[0])**2 + (c1[1]-p0[1])**2) + ((c0[0]-p1[0])**2 + (c0[1]-p1[1])**2)

                if cost_swapped < cost_direct:
                    sorted_hands = [h_curr[1], h_curr[0]]
                else:
                    sorted_hands = [h_curr[0], h_curr[1]]
            elif len(self._prev_hand_centers) == 1:
                p0 = self._prev_hand_centers[0]
                c0 = get_hand_pos(h_curr[0])
                c1 = get_hand_pos(h_curr[1])
                d0 = (c0[0]-p0[0])**2 + (c0[1]-p0[1])**2
                d1 = (c1[0]-p0[0])**2 + (c1[1]-p0[1])**2
                if d1 < d0:
                    sorted_hands = [h_curr[1], h_curr[0]]
                else:
                    sorted_hands = [h_curr[0], h_curr[1]]
            else:
                sorted_hands = sorted(h_curr, key=lambda h: min(float(h[4].x), float(h[8].x)) if (h is not None and len(h) >= 21) else 0.0)

            self._prev_hand_centers = [get_hand_pos(sorted_hands[0]), get_hand_pos(sorted_hands[1])]
        else:
            curr_hand = landmarks_list[0]
            if len(self._prev_hand_centers) == 2:
                p0, p1 = self._prev_hand_centers
                c = get_hand_pos(curr_hand)
                d0 = (c[0]-p0[0])**2 + (c[1]-p0[1])**2
                d1 = (c[0]-p1[0])**2 + (c[1]-p1[1])**2
                if d1 < d0:
                    # Previous hand 0 vanished, only hand 1 remains
                    self._f1_smoother.reset()
                self._f2_smoother.reset()

            sorted_hands = [curr_hand]
            self._prev_hand_centers = [get_hand_pos(sorted_hands[0])]

        # Gesture recognition on stably ordered hands with shape-mode isolation
        is_shaping = (num_hands >= 2)
        if self.extra_gestures_enabled:
            self.gesture_recognizer.update(sorted_hands, now=now, shape_mode=is_shaping)
            active_gesture = self.gesture_recognizer.get_active_gesture_name()
        else:
            active_gesture = "box_only" if num_hands > 0 else "none"

        points = []
        thumb1, index1 = (0.0, 0.0), (0.0, 0.0)
        thumb2, index2 = (0.0, 0.0), (0.0, 0.0)

        for i, hand in enumerate(sorted_hands):
            if hand is None or len(hand) < 21:
                continue
            t_tip = hand[4]
            i_tip = hand[8]
            t_pt = (float(t_tip.x), float(t_tip.y))
            i_pt = (float(i_tip.x), float(i_tip.y))

            points.extend([t_pt, i_pt])
            if i == 0:
                thumb1, index1 = t_pt, i_pt
            elif i == 1:
                thumb2, index2 = t_pt, i_pt

        if not points:
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        if num_hands == 1:
            padding_x = 0.08
            padding_y = 0.08
            min_x = max(0.0, min_x - padding_x)
            max_x = min(1.0, max_x + padding_x)
            min_y = max(0.0, min_y - padding_y)
            max_y = min(1.0, max_y + padding_y)
            self._f2_smoother.reset()
        else:
            pad = 0.01
            min_x = max(0.0, min_x - pad)
            max_x = min(1.0, max_x + pad)
            min_y = max(0.0, min_y - pad)
            max_y = min(1.0, max_y + pad)

        min_size = 0.06
        if (max_x - min_x) < min_size:
            center_x = (min_x + max_x) * 0.5
            min_x = max(0.0, center_x - min_size * 0.5)
            max_x = min(1.0, center_x + min_size * 0.5)
        if (max_y - min_y) < min_size:
            center_y = (min_y + max_y) * 0.5
            min_y = max(0.0, center_y - min_size * 0.5)
            max_y = min(1.0, center_y + min_size * 0.5)

        raw_box = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)
        smoothed_box = self._box_smoother.update(raw_box)

        f1 = self._f1_smoother.update(np.array([thumb1[0], thumb1[1], index1[0], index1[1]], dtype=np.float32))
        f2 = self._f2_smoother.update(np.array([thumb2[0], thumb2[1], index2[0], index2[1]], dtype=np.float32)) if num_hands > 1 else np.zeros(4, dtype=np.float32)

        with self._lock:
            self._state = TrackingData(
                has_box=1.0,
                box=(float(smoothed_box[0]), float(smoothed_box[1]), float(smoothed_box[2]), float(smoothed_box[3])),
                hands_count=num_hands,
                thumb1=(float(f1[0]), float(f1[1])),
                index1=(float(f1[2]), float(f1[3])),
                thumb2=(float(f2[0]), float(f2[1])),
                index2=(float(f2[2]), float(f2[3])),
                inference_time_ms=inference_ms,
                tracking_fps=self._track_fps,
                active_gesture=active_gesture,
                boost_active=self.boost_mode,
                wink_detected=self._wink_event
            )

    def poll_gesture_events(self):
        if not self.extra_gestures_enabled:
            return []
        return self.gesture_recognizer.poll_events()

    def get_continuous_gesture(self):
        if not self.extra_gestures_enabled:
            return None
        return self.gesture_recognizer.get_continuous_gesture()

    def poll_wink_event(self) -> bool:
        with self._lock:
            if self._wink_event:
                self._wink_event = False
                return True
            return False

    def get_state(self) -> TrackingData:
        with self._lock:
            return self._state

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.face_tracker:
            try:
                self.face_tracker.close()
            except Exception:
                pass

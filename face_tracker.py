"""
Face tracking and eye wink detection module using MediaPipe FaceLandmarker blendshapes.
"""

import os
import time
import urllib.request
from typing import Optional, Tuple
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_FILENAME = "face_landmarker.task"


class WinkDetector:
    """Detects deliberate eye winks (one eye closed, opposite eye open) while rejecting bilateral blinks."""

    def __init__(self, cooldown: float = 0.85, min_frames: int = 2):
        self.cooldown = cooldown
        self.min_frames = min_frames
        self.last_wink_time = -10.0
        self.wink_frames = 0

    def update(self, blink_left: float, blink_right: float, now: float) -> bool:
        if (now - self.last_wink_time) < self.cooldown:
            self.wink_frames = 0
            return False

        # Bilateral blink rejection: both eyes closing simultaneously is a blink, not a wink
        if blink_left > 0.38 and blink_right > 0.38:
            self.wink_frames = 0
            return False

        # Unilateral eye closure
        is_left_wink = (blink_left >= 0.45 and blink_right <= 0.28)
        is_right_wink = (blink_right >= 0.45 and blink_left <= 0.28)

        if (is_left_wink or is_right_wink) and abs(blink_left - blink_right) >= 0.28:
            self.wink_frames += 1
            if self.wink_frames >= self.min_frames:
                self.last_wink_time = now
                self.wink_frames = 0
                return True
        else:
            self.wink_frames = 0

        return False


class FaceTracker:
    """Tracks face landmarks and blendshapes to detect winks in real time."""

    def __init__(self, model_path: Optional[str] = None, cooldown: float = 0.85):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), FACE_MODEL_FILENAME)
        self.detector = None
        self.wink_detector = WinkDetector(cooldown=cooldown)
        self.blink_left = 0.0
        self.blink_right = 0.0
        self.face_present = False
        self._init_detector()

    def _ensure_model_exists(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"[FaceTracker] Model not found at {self.model_path}. Downloading from Google...")
            try:
                urllib.request.urlretrieve(FACE_MODEL_URL, self.model_path)
                print(f"[FaceTracker] Download complete ({os.path.getsize(self.model_path)} bytes).")
            except Exception as e:
                print(f"[FaceTracker] Failed to download face model: {e}")
                return False
        return True

    def _init_detector(self):
        if not MEDIAPIPE_AVAILABLE:
            print("[FaceTracker] MediaPipe unavailable. Face tracking disabled.")
            return

        if not self._ensure_model_exists():
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True
            )
            self.detector = mp_vision.FaceLandmarker.create_from_options(options)
            print("[FaceTracker] MediaPipe FaceLandmarker initialized successfully.")
        except Exception as e:
            print(f"[FaceTracker] Error initializing FaceLandmarker: {e}")
            self.detector = None

    def process_frame(self, rgb_small: np.ndarray, now: Optional[float] = None) -> bool:
        """Process an RGB frame and return True if a wink was detected."""
        if now is None:
            now = time.perf_counter()

        if self.detector is None:
            return False

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            result = self.detector.detect(mp_image)
        except Exception:
            return False

        if not result or not result.face_blendshapes or len(result.face_blendshapes) == 0:
            self.face_present = False
            self.blink_left = 0.0
            self.blink_right = 0.0
            return False

        self.face_present = True
        blendshapes = result.face_blendshapes[0]

        b_left = 0.0
        b_right = 0.0
        for category in blendshapes:
            name = category.category_name
            if name == "eyeBlinkLeft":
                b_left = float(category.score)
            elif name == "eyeBlinkRight":
                b_right = float(category.score)

        self.blink_left = b_left
        self.blink_right = b_right

        return self.wink_detector.update(b_left, b_right, now)

    def close(self):
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

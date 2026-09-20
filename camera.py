import time
import threading
import numpy as np
import cv2


class CameraCapture:


    def __init__(self, camera_index=0, preferred_width=1280, preferred_height=720, mirrored=True, mock=False):
        self.camera_index = camera_index
        self.preferred_width = preferred_width
        self.preferred_height = preferred_height
        self.mirrored = mirrored
        self.mock = mock

        self._cap = None
        self._thread = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()

        self._latest_frame = None
        self._frame_id = 0
        self._latest_timestamp = 0.0

        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
        self._dropped_frames = 0
        self._backend_name = "None"
        self._actual_width = 0
        self._actual_height = 0


    def start(self):
        if self._running:
            return

        self._running = True

        if not self.mock:
            self._init_camera()

        if self._cap is None or not self._cap.isOpened():
            print("[Camera] Physical camera unavailable. Falling back to synthetic mock stream.")
            self.mock = True
            self._backend_name = "Synthetic Mock Pattern"
            self._actual_width = 640
            self._actual_height = 480

        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="CameraCaptureWorker")
        self._thread.start()


    def _init_camera(self):
        backends = [
            ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
            ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
            ("Default (CAP_ANY)", cv2.CAP_ANY)
        ]


        for name, backend in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(self.camera_index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.preferred_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preferred_height)

                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        self._cap = cap
                        self._backend_name = name
                        self._actual_height, self._actual_width = test_frame.shape[:2]
                        print(f"[Camera] Connected via {name} at {self._actual_width}x{self._actual_height}")
                        return
                    else:
                        cap.release()
                else:
                    cap.release()
            except Exception as e:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                print(f"[Camera] Failed to initialize backend {name}: {e}")

        self._cap = None


    def _capture_worker(self):
        last_frame_id = -1
        consecutive_grab_failures = 0
        MAX_CONSECUTIVE_FAILURES = 30

        if not self.mock and self._cap:
            for _ in range(3):
                try:
                    self._cap.grab()
                except Exception:
                    pass

        while self._running:
            if self.mock:
                frame = self._generate_mock_frame()
                if self.mirrored:
                    frame = cv2.flip(frame, 1)
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_id += 1
                    self._latest_timestamp = time.perf_counter()
                    self._new_frame_event.set()
                time.sleep(1.0 / 60.0)
            else:
                grabbed = False
                if self._cap:
                    try:
                        grabbed = self._cap.grab()
                    except Exception:
                        grabbed = False

                if not grabbed:
                    consecutive_grab_failures += 1
                    if consecutive_grab_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"[Camera] {consecutive_grab_failures} consecutive grab failures. Falling back to synthetic mock stream.")
                        if self._cap:
                            try:
                                self._cap.release()
                            except Exception:
                                pass
                            self._cap = None
                        self.mock = True
                        self._backend_name = "Synthetic Mock Pattern (Fallback)"
                        self._actual_width = 640
                        self._actual_height = 480
                    time.sleep(0.002)
                    continue

                consecutive_grab_failures = 0
                ret, frame = False, None
                if self._cap:
                    try:
                        ret, frame = self._cap.retrieve()
                    except Exception:
                        ret, frame = False, None

                if not ret or frame is None:
                    time.sleep(0.002)
                    continue

                if self.mirrored:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_id += 1
                    self._latest_timestamp = now
                    self._new_frame_event.set()

            self._frame_count += 1
            elapsed = time.perf_counter() - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.perf_counter()



    def _generate_mock_frame(self):
        w, h = 640, 480
        t = time.perf_counter()
        img = np.zeros((h, w, 3), dtype=np.uint8)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        img[..., 0] = ((np.sin(x_coords * 0.01 + t * 2.0) + 1.0) * 80).astype(np.uint8)
        img[..., 1] = ((np.cos(y_coords * 0.01 + t * 1.5) + 1.0) * 80).astype(np.uint8)
        img[..., 2] = 120


        cx1 = int((np.sin(t * 1.2) * 0.35 + 0.5) * w)
        cy1 = int((np.cos(t * 1.4) * 0.35 + 0.5) * h)
        cv2.circle(img, (cx1, cy1), 40, (0, 255, 255), -1)

        cx2 = int((np.cos(t * 0.9) * 0.35 + 0.5) * w)
        cy2 = int((np.sin(t * 1.1) * 0.35 + 0.5) * h)
        cv2.circle(img, (cx2, cy2), 30, (255, 100, 50), -1)

        cv2.putText(img, "TEST PATTERN (NO CAMERA)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Time: {t:.2f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return img

    def get_latest_frame(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None, 0, 0.0
            return self._latest_frame, self._frame_id, self._latest_timestamp

    def wait_for_new_frame(self, timeout=0.03):
        return self._new_frame_event.wait(timeout)

    def consume_new_frame_event(self):
        self._new_frame_event.clear()

    def set_mirrored(self, mirrored: bool):
        self.mirrored = mirrored

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        print("[Camera] Stoped.")

    @property
    def fps(self):
        return self._fps

    @property
    def dropped_frames(self):
        return self._dropped_frames

    @property
    def resolution(self):
        if self._actual_width > 0 and self._actual_height > 0:
            return self._actual_width, self._actual_height
        return 640, 480

    @property
    def backend_name(self):
        return self._backend_name


import os
import cv2
import av
import numpy as np
import mediapipe as mp
import threading

from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS

# Push-up detector is optional until its file is available.
try:
    from detectors.pushup import PushUpDetector
except ImportError:
    PushUpDetector = None


class VideoProcessorClass(VideoProcessorBase):

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"
        self._frame_timestamps_ms = 0

        model_path = os.path.join(
            os.getcwd(),
            "ml_models",
            "pose_landmarker_full.task"
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Pose model not found: {model_path}"
            )

        base_option = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False
        )

        self._landmarker = (
            vision.PoseLandmarker.create_from_options(options)
        )

        self._detectors = {
            "Squats": SquatDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        if PushUpDetector is not None:
            self._detectors["Push-ups"] = PushUpDetector()

    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            if self._latest_metrics is None:
                return None
            return self._latest_metrics.copy()

    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        # Draw green body skeleton connections.
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(landmarks):
                continue
            if end_idx >= len(landmarks):
                continue

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if p1.visibility > 0.7 and p2.visibility > 0.7:
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)

                if (
                    0 <= x1 < w and 0 <= y1 < h
                    and 0 <= x2 < w and 0 <= y2 < h
                ):
                    cv2.line(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA
                    )

        # Draw blue dots at visible body landmarks.
        for lm in landmarks:
            if lm.visibility > 0.7:
                x, y = int(lm.x * w), int(lm.y * h)

                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(
                        img,
                        (x, y),
                        5,
                        (255, 0, 0),
                        -1,
                        cv2.LINE_AA
                    )

    def _draw_no_pose_warnings(self, img):
        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def _draw_overlays(self, img, metrics, ex_type):
        h, _ = img.shape[:2]

        if ex_type == "Squats":
            text = f"DEPTH: {metrics.get('depth_status', 'N/A')}"

        elif ex_type == "Push-ups":
            body = metrics.get("body_alignment", "N/A")
            hip = metrics.get("hip_status", "N/A")
            text = f"BODY: {body} | HIP: {hip}"

        elif ex_type == "Biceps Curls (Dumbbell)":
            text = f"SWING: {metrics.get('swing_status', 'N/A')}"

        elif ex_type == "Shoulder Press":
            ext = metrics.get("extension_status", "N/A")
            back = metrics.get("back_arch_status", "N/A")
            text = f"EXT: {ext} | BACK: {back}"

        elif ex_type == "Lunges":
            text = f"BALANCE: {metrics.get('balance_status', 'N/A')}"

        else:
            return

        cv2.putText(
            img,
            text,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def recv(self, frame):
        # Streamlit WebRTC supplies a BGR frame.
        image = frame.to_ndarray(format="bgr24")
        # print("recv called",image.shape)
        # cv2.putText(image,"RECV OK",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        image = cv2.flip(image, 1)

        # MediaPipe SRGB mode requires RGB pixels.
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # VIDEO mode requires increasing timestamps.
        self._frame_timestamps_ms += 33

        result = self._landmarker.detect_for_video(
            mp_image,
            self._frame_timestamps_ms
        )

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Draw lines on the original BGR frame.
            self._draw_skeleton(image, landmarks)

            ex_type = self.get_exercise()
            detector = self._detectors.get(ex_type)

            if detector is not None:
                metrics = detector.process(landmarks)

                if metrics is not None:
                    metrics["pose_detected"] = True
                    self._draw_overlays(
                        image,
                        metrics,
                        ex_type
                    )
                    self.set_latest_metrics(metrics)

        else:
            self._draw_no_pose_warnings(image)

            with self._lock:
                if self._latest_metrics is None:
                    self._latest_metrics = {}

                self._latest_metrics["pose_detected"] = False

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

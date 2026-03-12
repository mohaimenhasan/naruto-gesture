import os

import cv2
import mediapipe as mp
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


class HandCapture:
    """Extracts hand landmarks from webcam frames using MediaPipe Tasks API."""

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions(
            model_asset_path=os.path.join(ASSETS_DIR, "hand_landmarker.task")
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    def extract_landmarks(self, frame):
        """Extract normalized hand landmarks from a BGR frame.

        Returns list of landmark arrays, one per detected hand.
        Each array has shape (63,) — 21 landmarks × 3 coords (x, y, z).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33  # ~30fps timestamps in ms
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        hands_data = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])
                hands_data.append(np.array(landmarks, dtype=np.float32))

        return hands_data, result

    def draw_landmarks(self, frame, result):
        """Draw hand landmarks on frame for visualization."""
        drawing_utils = mp.tasks.vision.drawing_utils
        hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Convert to NormalizedLandmarkList proto for drawing
                proto = mp.tasks.vision.HandLandmarker.result_to_proto(result)
                break

            # Draw manually using cv2
            h, w = frame.shape[:2]
            for hand_landmarks in result.hand_landmarks:
                points = []
                for lm in hand_landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append((px, py))
                    cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

                for conn in hand_connections:
                    cv2.line(frame, points[conn.start], points[conn.end], (0, 255, 0), 2)

        return frame

    def release(self):
        self.landmarker.close()


class FaceCapture:
    """Extracts face mesh landmarks for eye tracking (Sharingan overlay)."""

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions(
            model_asset_path=os.path.join(ASSETS_DIR, "face_landmarker.task")
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_ts = 0

    def get_eye_positions(self, frame):
        """Get left and right eye center positions in pixel coordinates."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.face_landmarks:
            return None, None

        face = result.face_landmarks[0]

        # Eye landmark indices (iris centers from refined landmarks)
        LEFT_IRIS = [468, 469, 470, 471, 472]
        RIGHT_IRIS = [473, 474, 475, 476, 477]

        def eye_center(indices):
            xs = [face[i].x * w for i in indices]
            ys = [face[i].y * h for i in indices]
            return int(np.mean(xs)), int(np.mean(ys))

        # Estimate eye size from eye contour for scaling
        LEFT_EYE_CONTOUR = [33, 133]  # inner and outer corners
        RIGHT_EYE_CONTOUR = [362, 263]

        def eye_radius(contour_indices):
            p1 = face[contour_indices[0]]
            p2 = face[contour_indices[1]]
            dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) * w
            return int(dist * 0.75)

        left_center = eye_center(LEFT_IRIS)
        right_center = eye_center(RIGHT_IRIS)
        left_radius = eye_radius(LEFT_EYE_CONTOUR)
        right_radius = eye_radius(RIGHT_EYE_CONTOUR)

        return (left_center, left_radius), (right_center, right_radius)

    def release(self):
        self.landmarker.close()

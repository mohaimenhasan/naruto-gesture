import cv2
import mediapipe as mp
import numpy as np


class HandCapture:
    """Extracts hand landmarks from webcam frames using MediaPipe."""

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_landmarks(self, frame):
        """Extract normalized hand landmarks from a BGR frame.

        Returns list of landmark arrays, one per detected hand.
        Each array has shape (63,) — 21 landmarks × 3 coords (x, y, z).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                hands_data.append(np.array(landmarks, dtype=np.float32))

        return hands_data, results

    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on frame for visualization."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def release(self):
        self.hands.close()


class FaceCapture:
    """Extracts face mesh landmarks for eye tracking (Sharingan overlay)."""

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def get_eye_positions(self, frame):
        """Get left and right eye center positions in pixel coordinates."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        face = results.multi_face_landmarks[0]

        # Eye landmark indices (iris centers from refined landmarks)
        LEFT_IRIS = [468, 469, 470, 471, 472]
        RIGHT_IRIS = [473, 474, 475, 476, 477]

        def eye_center(indices):
            xs = [face.landmark[i].x * w for i in indices]
            ys = [face.landmark[i].y * h for i in indices]
            return int(np.mean(xs)), int(np.mean(ys))

        # Estimate eye size from eye contour for scaling
        LEFT_EYE_CONTOUR = [33, 133]  # inner and outer corners
        RIGHT_EYE_CONTOUR = [362, 263]

        def eye_radius(contour_indices):
            p1 = face.landmark[contour_indices[0]]
            p2 = face.landmark[contour_indices[1]]
            dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) * w
            return int(dist * 0.75)

        left_center = eye_center(LEFT_IRIS)
        right_center = eye_center(RIGHT_IRIS)
        left_radius = eye_radius(LEFT_EYE_CONTOUR)
        right_radius = eye_radius(RIGHT_EYE_CONTOUR)

        return (left_center, left_radius), (right_center, right_radius)

    def release(self):
        self.face_mesh.close()

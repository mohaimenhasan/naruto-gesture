import os
import pickle

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")


class GestureClassifier:
    """Predicts gestures from hand landmarks using a trained model."""

    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.encoder = None
        self._load_model()

    def _load_model(self):
        if not os.path.isfile(MODEL_PATH):
            print("No trained model found. Run training mode first.")
            return

        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            self.encoder = pickle.load(f)

        print(f"Loaded model with gestures: {list(self.encoder.classes_)}")

    def predict(self, landmarks):
        """Predict gesture from landmarks array.

        Returns (gesture_name, confidence) or (None, 0.0) if below threshold.
        """
        if self.model is None:
            return None, 0.0

        X = landmarks.reshape(1, -1)
        probas = self.model.predict_proba(X)[0]
        max_idx = np.argmax(probas)
        confidence = probas[max_idx]

        if confidence >= self.confidence_threshold:
            gesture = self.encoder.inverse_transform([max_idx])[0]
            return gesture, confidence

        return None, confidence

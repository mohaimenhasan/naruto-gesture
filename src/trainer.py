import csv
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GESTURE_CSV = os.path.join(DATA_DIR, "gestures.csv")
MODEL_PATH = os.path.join(DATA_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")

GESTURE_KEYS = {
    ord("s"): "sharingan",
    ord("c"): "chidori",
    ord("r"): "rasengan",
}


def save_sample(landmarks, label):
    """Append a landmark sample with its label to the CSV dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.isfile(GESTURE_CSV)

    with open(GESTURE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"lm_{i}" for i in range(len(landmarks))] + ["label"]
            writer.writerow(header)
        row = landmarks.tolist() + [label]
        writer.writerow(row)


def load_dataset():
    """Load gesture dataset from CSV."""
    if not os.path.isfile(GESTURE_CSV):
        return None, None

    data, labels = [], []
    with open(GESTURE_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            data.append([float(x) for x in row[:-1]])
            labels.append(row[-1])

    return np.array(data), np.array(labels)


def train_model():
    """Train a gesture classifier and save it to disk."""
    X, y = load_dataset()
    if X is None or len(X) < 5:
        print("Not enough training data. Collect more samples.")
        return None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Model trained! Accuracy: {accuracy:.1%}")
    print(f"Classes: {list(encoder.classes_)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    return clf

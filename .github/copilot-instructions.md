# Copilot Instructions for naruto-gesture

## Project Overview

A local Naruto-themed gesture recognition app. Users train custom hand gestures via webcam, then perform them in real-time to trigger full-screen jutsu visual effects (Sharingan, Chidori, Rasengan).

## Tech Stack

- **Python 3** with **MediaPipe** (hand landmarks + face mesh), **OpenCV** (camera/image processing), **Pygame** (full-screen effects), **scikit-learn** (gesture classification)

## Architecture

Two modes driven from `src/app.py`:

- **Training mode** (`--mode train`): OpenCV window captures webcam frames → `src/capture.HandCapture` extracts 21 hand landmarks (63 floats) → user presses a key to label the gesture → `src/trainer` saves landmarks+label to `data/gestures.csv` and trains a `RandomForestClassifier` saved to `data/gesture_model.pkl`.
- **Battle mode** (`--mode battle`): Pygame fullscreen loop captures webcam frames → `src/capture.HandCapture` extracts landmarks → `src/classifier.GestureClassifier` predicts the gesture → the matching effect renderer in `src/effects/` draws over the camera feed. Sharingan uses `src/capture.FaceCapture` (MediaPipe FaceMesh) to track eye positions and overlay the image. Chidori and Rasengan are procedural particle/animation effects rendered with Pygame.

### Key data flow

```
Camera → HandCapture.extract_landmarks() → GestureClassifier.predict()
  → "sharingan" → FaceCapture.get_eye_positions() → SharinganEffect.render(frame, eyes)
  → "chidori"   → ChidoriEffect.update(hand_pos) → ChidoriEffect.render(surface)
  → "rasengan"  → RasenganEffect.update(hand_pos) → RasenganEffect.render(surface)
```

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train gestures (collect ~30+ samples per gesture)
python -m src.app --mode train

# Battle mode
python -m src.app --mode battle
```

## Conventions

- All source code lives under `src/`; effects are in `src/effects/`.
- Training data and model artifacts go in `data/` (gitignored).
- Asset images (e.g., `sharingan.png`) go in `assets/`. Effects fall back to procedural generation if no asset exists.
- Hand landmarks are always a flat array of 63 floats (21 landmarks × 3 coords).
- Effects follow a consistent interface: `update()` to advance state, `render(surface)` to draw (Chidori/Rasengan), or `render(frame, eye_data)` for camera-overlay effects (Sharingan).
- Gesture key mapping is defined in `src/trainer.GESTURE_KEYS`. Add new jutsus there first.

## Adding a New Jutsu

1. Add its key mapping in `src/trainer.GESTURE_KEYS`
2. Create `src/effects/<jutsu_name>.py` with `update()` and `render()` methods
3. Wire it into the battle mode loop in `src/app.py`
4. Retrain the model with the new gesture samples

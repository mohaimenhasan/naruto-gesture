# Copilot Instructions for naruto-gesture

## Project Overview

A local Naruto-themed gesture recognition app. Users perform hand gestures and facial expressions via webcam to trigger full-screen jutsu visual effects (Sharingan, Rasengan).

## Tech Stack

- **Python 3** with **MediaPipe** (hand landmarks + face mesh), **OpenCV** (camera/image processing), **Pygame** (full-screen effects)

## Architecture

Single mode driven from `src/app.py`:

Pygame fullscreen loop captures webcam frames → `src/capture.HandCapture` extracts 21 hand landmarks and detects open-palm gestures using 3D distance heuristics → matching effect renderer in `src/effects/` draws over the camera feed. Sharingan uses `src/capture.FaceCapture` (MediaPipe FaceMesh) to track eye positions and detect blinks via EAR (Eye Aspect Ratio) with auto-calibration. Rasengan is a procedural particle/animation effect rendered with Pygame.

### Key data flow

```
Camera → HandCapture.extract_landmarks() → open-palm detection (3D distance from wrist)
  → palm open   → RasenganEffect.update(hand_pos) → RasenganEffect.render(surface)
  → eyes closed → FaceCapture.get_eye_positions()  → SharinganEffect.render(frame, eyes)
```

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.app
```

## Conventions

- All source code lives under `src/`; effects are in `src/effects/`.
- Asset images (e.g., `sharingan.png`) go in `assets/`. Effects fall back to procedural generation if no asset exists.
- Hand landmarks are always a flat array of 63 floats (21 landmarks × 3 coords).
- Effects follow a consistent interface: `update()` to advance state, `render(surface)` to draw (Rasengan), or `render(frame, eye_data)` for camera-overlay effects (Sharingan).
- Gesture detection is done via heuristic checks on MediaPipe landmarks (no ML model needed).

## Adding a New Jutsu

1. Create `src/effects/<jutsu_name>.py` with `update()` and `render()` methods
2. Add a detection heuristic in the main loop in `src/app.py`
3. Wire the effect into the render section of `src/app.py`

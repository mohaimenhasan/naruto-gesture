# Naruto Gesture Recognition

Perform Naruto hand signs in front of your webcam to trigger full-screen jutsu effects!

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### 1. Train Your Gestures

```bash
python -m src.app --mode train
```

- Show your hand gesture to the camera
- Press **s** to label it as **Sharingan**
- Press **c** to label it as **Chidori**
- Press **r** to label it as **Rasengan**
- Press **q** when done collecting samples
- The classifier trains automatically when you quit

Collect ~30+ samples per gesture for good accuracy.

### 2. Battle Mode

```bash
python -m src.app --mode battle
```

- Perform a trained gesture to trigger its jutsu effect
- **Sharingan** — overlays Sharingan on your eyes
- **Chidori** — full-screen lightning effect
- **Rasengan** — spinning energy sphere
- Press **q** to quit

## Supported Jutsus

| Gesture Key | Jutsu | Effect |
|-------------|-------|--------|
| S | Sharingan | Red Sharingan overlaid on your eyes |
| C | Chidori | Blue-white lightning crackling across screen |
| R | Rasengan | Spinning blue energy sphere |

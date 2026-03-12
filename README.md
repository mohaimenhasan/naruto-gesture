# Naruto Gesture Recognition

Perform Naruto jutsu with your webcam! Open your palm to build a Rasengan, close your eyes to activate Sharingan.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
python -m src.app
```

- **Open palm** → Rasengan builds on your hand (keeps growing while held!)
- **Close eyes 2s+ then reopen** → Toggles Sharingan on your eyes
- Both can be active simultaneously
- Press **Q** or **ESC** to quit

Blink detection auto-calibrates during the first 5 seconds — just use it naturally.

# Naruto Gesture Recognition

Perform Naruto jutsu with your webcam! Open your palm to summon a Rasengan, close your eyes to activate Sharingan.

## 🌐 Try it in your browser

**[Launch Web App](https://mohaimenhasan.github.io/naruto-gesture/)** — no install needed, works on desktop Chrome/Edge/Firefox with a webcam.

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

The app auto-calibrates during the first ~5 seconds — just look at the camera naturally.

### Jutsus

| Gesture | Effect |
|---------|--------|
| **Open palm** | Rasengan appears above your hand — grows while held, shrinks and fades when you lower it |
| **Close eyes ~1 s then reopen** | Toggles Sharingan overlay on your eyes |

Both effects can be active simultaneously. Press **Q** or **ESC** to quit.

## Desktop Setup (Python)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.app
```

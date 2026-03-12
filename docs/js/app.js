import { initVision, HandCapture, FaceCapture } from "./capture.js";
import { RasenganEffect } from "./rasengan.js";
import { SharinganEffect } from "./sharingan.js";

// DOM elements
const loadingScreen = document.getElementById("loading-screen");
const loadingStatus = document.getElementById("loading-status");
const startBtn = document.getElementById("start-btn");
const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const hudEl = document.getElementById("hud");
const instructionsEl = document.getElementById("instructions");
const ctx = canvas.getContext("2d");

// State
let handCapture, faceCapture, sharingan, rasengan;
let sharinganActive = false;
let rasenganActive = false;

// Blink detection state
let eyesClosedFrames = 0;
const BLINK_HOLD_FRAMES = 30;
let eyesWereHeld = false;
let sharinganCooldown = 0;
let frameCount = 0;

// Auto-calibration
const calibrationEars = [];
const CALIBRATION_FRAMES = 150;
let calibrated = false;

// Rasengan palm grace period
let palmLostFrames = 0;
const PALM_GRACE_FRAMES = 8;

// ─── Initialization ────────────────────────────────────────────

async function init() {
  const setStatus = (msg) => { loadingStatus.textContent = msg; };

  await initVision(setStatus);

  handCapture = new HandCapture();
  await handCapture.init(setStatus);

  faceCapture = new FaceCapture();
  await faceCapture.init(setStatus);

  setStatus("Loading assets…");
  sharingan = new SharinganEffect(1, 1);
  await sharingan.loadImage("assets/sharingan.png");

  setStatus("Ready! Click to start.");
  document.querySelector(".spinner").style.display = "none";
  startBtn.style.display = "inline-block";
  startBtn.addEventListener("click", startCamera);
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // Set canvas to actual video resolution
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Re-init effects with correct dimensions
    const w = canvas.width;
    const h = canvas.height;
    sharingan.width = w;
    sharingan.height = h;
    rasengan = new RasenganEffect(w, h);

    loadingScreen.classList.add("hidden");

    // Fade instructions after 8 seconds
    setTimeout(() => instructionsEl.classList.add("fade-out"), 8000);

    requestAnimationFrame(loop);
  } catch (err) {
    loadingStatus.textContent = `Camera error: ${err.message}`;
  }
}

// ─── Helper: 3D distance between two landmarks ────────────────

function dist3d(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

// ─── Main loop ─────────────────────────────────────────────────

let lastTimestamp = 0;
let processing = false;

function loop(timestamp) {
  requestAnimationFrame(loop);

  if (timestamp - lastTimestamp < 33) return;
  lastTimestamp = timestamp;

  // Guard: skip if previous async detection is still running
  if (processing) return;
  processing = true;
  processFrame().finally(() => { processing = false; });
}

async function processFrame() {
  const w = canvas.width;
  const h = canvas.height;

  // Draw mirrored video
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  // ── Face tracking + blink detection ──────────────────────

  const faceResult = await faceCapture.getEyePositions(video);
  let { leftEye, rightEye, eyesClosed } = faceResult;

  // Mirror the eye positions since we flipped the video
  if (leftEye) {
    leftEye = { center: [w - leftEye.center[0], leftEye.center[1]], radius: leftEye.radius };
  }
  if (rightEye) {
    rightEye = { center: [w - rightEye.center[0], rightEye.center[1]], radius: rightEye.radius };
  }

  // Auto-calibration
  if (!calibrated && leftEye) {
    calibrationEars.push(faceCapture._lastEar);
    if (frameCount >= CALIBRATION_FRAMES && calibrationEars.length > 30) {
      const sorted = [...calibrationEars].sort((a, b) => a - b);
      const n = sorted.length;
      const avgOpen = sorted.slice(Math.floor(n * 0.7)).reduce((s, v) => s + v, 0) /
        sorted.slice(Math.floor(n * 0.7)).length;
      const closedSlice = sorted.slice(0, Math.max(1, Math.floor(n * 0.1)));
      const avgClosed = closedSlice.reduce((s, v) => s + v, 0) / closedSlice.length;
      faceCapture.earThreshold = (avgOpen + avgClosed) / 2;
      calibrated = true;
    }
  }
  frameCount++;

  // Blink-to-toggle Sharingan
  if (sharinganCooldown > 0) sharinganCooldown--;

  if (eyesClosed) {
    eyesClosedFrames++;
    if (eyesClosedFrames >= BLINK_HOLD_FRAMES) eyesWereHeld = true;
  } else {
    if (eyesWereHeld && sharinganCooldown <= 0) {
      sharinganActive = !sharinganActive;
      sharinganCooldown = 60;
      if (sharinganActive) sharingan.reset();
    }
    eyesClosedFrames = 0;
    eyesWereHeld = false;
  }

  // ── Hand detection: open palm triggers Rasengan ──────────

  const handResult = await handCapture.detect(video);
  let palmDetected = false;
  let rasenganPos = null;
  let fingersOpen = 0;
  let thumbOpen = false;

  if (handResult && handResult.landmarks && handResult.landmarks.length > 0) {
    const lm = handResult.landmarks[0]; // array of {x, y, z}
    const wrist = lm[0];

    const tipIds = [8, 12, 16, 20];
    const dipIds = [6, 10, 14, 18];
    fingersOpen = 0;
    for (let i = 0; i < tipIds.length; i++) {
      if (dist3d(lm[tipIds[i]], wrist) > dist3d(lm[dipIds[i]], wrist)) fingersOpen++;
    }
    thumbOpen = dist3d(lm[4], wrist) > dist3d(lm[3], wrist);

    if (fingersOpen >= 3 && thumbOpen) {
      palmDetected = true;
      const palmX = (lm[9].x + lm[5].x + lm[17].x) / 3;
      const palmY = (lm[9].y + lm[5].y + lm[17].y) / 3;
      const PALM_OFFSET_Y = -h * 0.12;
      // Mirror x coordinate
      rasenganPos = [w - palmX * w, palmY * h + PALM_OFFSET_Y];
    }

    // Draw hand landmarks (debug)
    ctx.save();
    for (let i = 0; i < lm.length; i++) {
      const px = w - lm[i].x * w; // mirrored
      const py = lm[i].y * h;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = [4, 8, 12, 16, 20].includes(i) ? "#0f0" : "#ff0";
      ctx.fill();
    }
    // Palm center marker
    const pcx = w - ((lm[9].x + lm[5].x + lm[17].x) / 3) * w;
    const pcy = ((lm[9].y + lm[5].y + lm[17].y) / 3) * h;
    ctx.beginPath();
    ctx.arc(pcx, pcy, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#f00";
    ctx.fill();
    ctx.restore();

    // Detection info
    ctx.save();
    ctx.font = "16px monospace";
    ctx.fillStyle = "#0f0";
    ctx.fillText(
      `Fingers: ${fingersOpen}/4  Thumb: ${thumbOpen ? "Y" : "N"}  Palm: ${palmDetected ? "YES" : "NO"}`,
      10, h - 30
    );
    ctx.restore();
  } else {
    ctx.save();
    ctx.font = "16px monospace";
    ctx.fillStyle = "#f00";
    ctx.fillText("No hand detected", 10, h - 30);
    ctx.restore();
  }

  // ── Rasengan state machine ───────────────────────────────

  if (palmDetected && calibrated) {
    palmLostFrames = 0;
    if (!rasenganActive) {
      rasengan.reset();
      rasengan.alive = true;
      rasenganActive = true;
    }
    if (rasengan.fading) rasengan.fading = false;
    rasengan.update(rasenganPos);
  } else {
    if (rasenganActive) {
      palmLostFrames++;
      if (palmLostFrames >= PALM_GRACE_FRAMES && !rasengan.fading) rasengan.startFading();
      if (rasengan.fading) rasengan.update();
      if (!rasengan.alive) {
        rasenganActive = false;
        rasengan.reset();
        palmLostFrames = 0;
      }
    }
  }

  // ── Draw tracking outlines during calibration ────────────

  if (!calibrated && leftEye) {
    ctx.save();
    ctx.strokeStyle = "#0ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(leftEye.center[0], leftEye.center[1], leftEye.radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(rightEye.center[0], rightEye.center[1], rightEye.radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // ── Sharingan eye overlay ────────────────────────────────

  if (sharinganActive && leftEye) {
    sharingan.render(ctx, { leftEye, rightEye }, eyesClosed);
  }

  // ── Rasengan on top ──────────────────────────────────────

  if (rasenganActive) {
    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
    rasengan.render(ctx);
  }

  // ── HUD ──────────────────────────────────────────────────

  const hudLines = [];
  if (!calibrated) {
    const pct = Math.min((frameCount / CALIBRATION_FRAMES) * 100, 100);
    hudLines.push({ text: `CALIBRATING: ${pct.toFixed(0)}%`, cls: "calibrating" });
  }
  if (eyesClosed && eyesClosedFrames > 0) {
    const progress = Math.min((eyesClosedFrames / BLINK_HOLD_FRAMES) * 100, 100);
    hudLines.push({ text: `SHARINGAN CHARGING: ${progress.toFixed(0)}%`, cls: "charging" });
  }
  if (sharinganActive) {
    hudLines.push({ text: "SHARINGAN: ACTIVE", cls: "sharingan" });
  }
  if (rasenganActive) {
    const sizePct = Math.floor((rasengan.baseRadius / RasenganEffect.MAX_RADIUS) * 100);
    hudLines.push({ text: `RASENGAN: ${sizePct}% POWER`, cls: "rasengan" });
  }

  hudEl.innerHTML = hudLines
    .map(l => `<div class="hud-line ${l.cls}">${l.text}</div>`)
    .join("");
}

// ── Keyboard shortcut ──────────────────────────────────────────

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    video.srcObject?.getTracks().forEach(t => t.stop());
    location.reload();
  }
});

// ── Start ──────────────────────────────────────────────────────

init().catch((err) => {
  console.error(err);
  loadingStatus.innerHTML =
    `<strong>Error:</strong> ${err.message}<br><br>` +
    `<strong>Troubleshooting:</strong><br>` +
    `1. <strong>Restart your browser</strong> (close ALL windows, then reopen)<br>` +
    `2. Enable Hardware Acceleration: <code>chrome://settings/system</code><br>` +
    `3. Check WebGL status: <code>chrome://gpu</code><br>` +
    `4. Try a different browser (Chrome, Edge, or Firefox)`;
  loadingStatus.style.textAlign = "left";
  loadingStatus.style.fontSize = "0.9rem";
  loadingStatus.style.maxWidth = "500px";
  document.querySelector(".spinner").style.display = "none";
});

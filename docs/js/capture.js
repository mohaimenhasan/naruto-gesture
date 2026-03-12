// MediaPipe Tasks Vision API — single shared WASM/WebGL instance for both
// hand and face detection.  Frames are captured to an intermediate canvas
// before being passed to MediaPipe so the video element itself never touches
// WebGL (avoids driver-level texture-upload bugs).

import { FilesetResolver, HandLandmarker, FaceLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const NO_FACE = { leftEye: null, rightEye: null, eyesClosed: false };

// Shared vision fileset — one WASM module, one WebGL context
let _vision = null;

export async function initVision(onStatus) {
  onStatus?.("Downloading MediaPipe WASM runtime…");
  _vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  return _vision;
}

// ─── Temp canvas for frame capture ─────────────────────────────
// Drawing the video to a plain 2D canvas first, then handing that canvas
// to MediaPipe avoids the problematic HTMLVideoElement → WebGL texture path
// that crashes on some browsers/drivers.

let _tmpCanvas = null;
let _tmpCtx = null;

export function captureFrame(video) {
  if (!_tmpCanvas) {
    _tmpCanvas = document.createElement("canvas");
    _tmpCtx = _tmpCanvas.getContext("2d");
  }
  if (_tmpCanvas.width !== video.videoWidth || _tmpCanvas.height !== video.videoHeight) {
    _tmpCanvas.width = video.videoWidth;
    _tmpCanvas.height = video.videoHeight;
  }
  _tmpCtx.drawImage(video, 0, 0);
  return _tmpCanvas;
}

// ─── Hand capture ──────────────────────────────────────────────

export class HandCapture {
  constructor() {
    this.landmarker = null;
    this._lastTs = -1;
  }

  async init(onStatus) {
    onStatus?.("Loading hand detection model…");
    const opts = {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    };
    // Try GPU first, then CPU
    for (const delegate of ["GPU", "CPU"]) {
      try {
        opts.baseOptions.delegate = delegate;
        this.landmarker = await HandLandmarker.createFromOptions(_vision, opts);
        console.log(`HandLandmarker ready (${delegate})`);
        return;
      } catch (e) {
        console.warn(`HandLandmarker ${delegate} init failed:`, e.message);
      }
    }
    throw new Error("Hand model failed to load. See troubleshooting below.");
  }

  detect(frameCanvas, timestampMs) {
    if (!this.landmarker) return null;
    const ts = Math.max(Math.round(timestampMs), this._lastTs + 1);
    this._lastTs = ts;
    try {
      return this.landmarker.detectForVideo(frameCanvas, ts);
    } catch (e) {
      console.warn("Hand detection error:", e);
      return null;
    }
  }
}

// ─── Face capture ──────────────────────────────────────────────

export class FaceCapture {
  constructor() {
    this.landmarker = null;
    this.earThreshold = 0.30;
    this._lastEar = 0;
    this._lastTs = -1;
  }

  async init(onStatus) {
    onStatus?.("Loading face detection model…");
    const opts = {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    };
    for (const delegate of ["GPU", "CPU"]) {
      try {
        opts.baseOptions.delegate = delegate;
        this.landmarker = await FaceLandmarker.createFromOptions(_vision, opts);
        console.log(`FaceLandmarker ready (${delegate})`);
        return;
      } catch (e) {
        console.warn(`FaceLandmarker ${delegate} init failed:`, e.message);
      }
    }
    throw new Error("Face model failed to load. See troubleshooting below.");
  }

  getEyePositions(frameCanvas, timestampMs) {
    if (!this.landmarker) return NO_FACE;

    const ts = Math.max(Math.round(timestampMs), this._lastTs + 1);
    this._lastTs = ts;

    let result;
    try {
      result = this.landmarker.detectForVideo(frameCanvas, ts);
    } catch (e) {
      console.warn("Face detection error:", e);
      return NO_FACE;
    }

    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      return NO_FACE;
    }

    const face = result.faceLandmarks[0];
    const w = frameCanvas.width;
    const h = frameCanvas.height;

    // EAR (Eye Aspect Ratio) for blink detection
    const earMulti = (upper, lower, corners) => {
      let vertSum = 0;
      for (let i = 0; i < upper.length; i++) {
        const dx = face[upper[i]].x - face[lower[i]].x;
        const dy = face[upper[i]].y - face[lower[i]].y;
        vertSum += Math.sqrt(dx * dx + dy * dy);
      }
      const avgVert = vertSum / upper.length;
      const cdx = face[corners[0]].x - face[corners[1]].x;
      const cdy = face[corners[0]].y - face[corners[1]].y;
      const horiz = Math.sqrt(cdx * cdx + cdy * cdy);
      return avgVert / (horiz + 1e-6);
    };

    const leftEar = earMulti([159, 160, 161], [144, 145, 153], [33, 133]);
    const rightEar = earMulti([386, 385, 384], [373, 374, 380], [263, 362]);
    const avgEar = (leftEar + rightEar) / 2;
    this._lastEar = avgEar;
    const eyesClosed = avgEar < this.earThreshold;

    const eyeCenter = (indices) => {
      let sx = 0, sy = 0;
      for (const i of indices) {
        sx += face[i].x * w;
        sy += face[i].y * h;
      }
      return [Math.round(sx / indices.length), Math.round(sy / indices.length)];
    };

    const irisRadius = (indices) => {
      let cx = 0, cy = 0;
      for (const i of indices) { cx += face[i].x; cy += face[i].y; }
      cx /= indices.length; cy /= indices.length;
      let maxDist = 0;
      for (const i of indices) {
        const dx = face[i].x - cx, dy = face[i].y - cy;
        maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy));
      }
      return Math.round(maxDist * w * 1.2);
    };

    const LEFT_IRIS = [468, 469, 470, 471, 472];
    const RIGHT_IRIS = [473, 474, 475, 476, 477];

    return {
      leftEye: { center: eyeCenter(LEFT_IRIS), radius: irisRadius(LEFT_IRIS) },
      rightEye: { center: eyeCenter(RIGHT_IRIS), radius: irisRadius(RIGHT_IRIS) },
      eyesClosed,
    };
  }
}

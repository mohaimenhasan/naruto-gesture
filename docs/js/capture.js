// TensorFlow.js with WASM backend — runs entirely on CPU, NO WebGL needed.
// Uses the same MediaPipe model architectures via TF.js model packages.
// Globals loaded by <script> tags: tf, handPoseDetection, faceLandmarksDetection

const NO_FACE = { leftEye: null, rightEye: null, eyesClosed: false };

export async function initVision(onStatus) {
  onStatus?.("Initializing AI runtime (WASM)…");
  const tf = window.tf;
  if (!tf) throw new Error("TensorFlow.js failed to load. Please refresh the page.");

  tf.wasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tf.version_core}/wasm-out/`
  );
  await tf.setBackend("wasm");
  await tf.ready();
  console.log(`TF.js ready — backend: ${tf.getBackend()}, v${tf.version_core}`);
}

export class HandCapture {
  constructor() {
    this._detector = null;
  }

  async init(onStatus) {
    onStatus?.("Loading hand detection model…");
    const hpd = window.handPoseDetection;
    this._detector = await hpd.createDetector(
      hpd.SupportedModels.MediaPipeHands,
      { runtime: "tfjs", maxHands: 1, modelType: "lite" }
    );
    console.log("Hand detector ready (TF.js WASM)");
  }

  async detect(video) {
    if (!this._detector) return null;
    try {
      const hands = await this._detector.estimateHands(video);
      if (!hands || hands.length === 0) return null;
      const w = video.videoWidth;
      const h = video.videoHeight;
      // Normalize to 0-1 coords so app.js gesture logic works unchanged
      const landmarks = hands.map((hand) =>
        hand.keypoints.map((kp, i) => ({
          x: kp.x / w,
          y: kp.y / h,
          z: hand.keypoints3D?.[i]?.z ?? 0,
        }))
      );
      return { landmarks };
    } catch (e) {
      console.warn("Hand detection error:", e);
      return null;
    }
  }
}

export class FaceCapture {
  constructor() {
    this._detector = null;
    this.earThreshold = 0.3;
    this._lastEar = 0;
  }

  async init(onStatus) {
    onStatus?.("Loading face detection model…");
    const fld = window.faceLandmarksDetection;
    this._detector = await fld.createDetector(
      fld.SupportedModels.MediaPipeFaceMesh,
      { runtime: "tfjs", maxFaces: 1, refineLandmarks: true }
    );
    console.log("Face detector ready (TF.js WASM)");
  }

  async getEyePositions(video) {
    if (!this._detector) return NO_FACE;
    let faces;
    try {
      faces = await this._detector.estimateFaces(video);
    } catch (e) {
      console.warn("Face detection error:", e);
      return NO_FACE;
    }
    if (!faces || faces.length === 0) return NO_FACE;

    // TF.js keypoints are in pixel coordinates
    const face = faces[0].keypoints;

    // EAR (Eye Aspect Ratio) — ratio is scale-invariant
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

    // Eye centers — already in pixel coords from TF.js
    const eyeCenter = (indices) => {
      let sx = 0, sy = 0;
      for (const i of indices) { sx += face[i].x; sy += face[i].y; }
      return [Math.round(sx / indices.length), Math.round(sy / indices.length)];
    };

    const irisRadius = (indices) => {
      let cx = 0, cy = 0;
      for (const i of indices) { cx += face[i].x; cy += face[i].y; }
      cx /= indices.length;
      cy /= indices.length;
      let maxDist = 0;
      for (const i of indices) {
        const dx = face[i].x - cx, dy = face[i].y - cy;
        maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy));
      }
      return Math.round(maxDist * 1.2);
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

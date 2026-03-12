// Uses the legacy MediaPipe Solution API (global Hands / FaceMesh) loaded
// via <script> tags in index.html.  These manage their own WebGL context
// reliably, unlike the newer Tasks-Vision WASM bundle.

export class HandCapture {
  constructor() {
    this._hands = null;
    this._latestResult = null;
    this._ready = false;
  }

  async init() {
    // eslint-disable-next-line no-undef
    this._hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });
    this._hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });
    this._hands.onResults((r) => { this._latestResult = r; });
    await this._hands.initialize();
    this._ready = true;
    console.log("HandCapture ready (legacy Solution API)");
  }

  async detect(video) {
    if (!this._ready) return null;
    try {
      await this._hands.send({ image: video });
    } catch (e) {
      console.warn("Hand detection frame error:", e);
      return null;
    }
    return this._latestResult;
  }
}

export class FaceCapture {
  constructor() {
    this._mesh = null;
    this._latestResult = null;
    this._ready = false;
    this.earThreshold = 0.30;
    this._lastEar = 0;
  }

  async init() {
    // eslint-disable-next-line no-undef
    this._mesh = new FaceMesh({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`,
    });
    this._mesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,   // enables iris landmarks 468-477
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });
    this._mesh.onResults((r) => { this._latestResult = r; });
    await this._mesh.initialize();
    this._ready = true;
    console.log("FaceCapture ready (legacy Solution API)");
  }

  async getEyePositions(video) {
    const NO_FACE = { leftEye: null, rightEye: null, eyesClosed: false };
    if (!this._ready) return NO_FACE;

    try {
      await this._mesh.send({ image: video });
    } catch (e) {
      console.warn("Face detection frame error:", e);
      return NO_FACE;
    }

    const result = this._latestResult;
    if (!result || !result.multiFaceLandmarks || result.multiFaceLandmarks.length === 0) {
      return NO_FACE;
    }

    const face = result.multiFaceLandmarks[0];
    const w = video.videoWidth;
    const h = video.videoHeight;

    // EAR (Eye Aspect Ratio) for blink detection — mirrors Python exactly
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

    // Iris centers (landmarks 468-472 left, 473-477 right)
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

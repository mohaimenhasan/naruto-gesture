import { FilesetResolver, HandLandmarker, FaceLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

export class HandCapture {
  constructor() {
    this.landmarker = null;
  }

  async init() {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    // Try GPU first, fall back to CPU if WebGL isn't available
    for (const delegate of ["GPU", "CPU"]) {
      try {
        this.landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            delegate,
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5,
        });
        console.log(`HandLandmarker using ${delegate} delegate`);
        return;
      } catch (e) {
        console.warn(`HandLandmarker ${delegate} failed, trying next…`, e);
      }
    }
    throw new Error("Could not initialize HandLandmarker on GPU or CPU");
  }

  detect(video, timestampMs) {
    if (!this.landmarker) return null;
    return this.landmarker.detectForVideo(video, timestampMs);
  }
}

export class FaceCapture {
  constructor() {
    this.landmarker = null;
    this.earThreshold = 0.30;
    this._lastEar = 0;
  }

  async init() {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    for (const delegate of ["GPU", "CPU"]) {
      try {
        this.landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            delegate,
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5,
        });
        console.log(`FaceLandmarker using ${delegate} delegate`);
        return;
      } catch (e) {
        console.warn(`FaceLandmarker ${delegate} failed, trying next…`, e);
      }
    }
    throw new Error("Could not initialize FaceLandmarker on GPU or CPU");
  }

  getEyePositions(video, timestampMs) {
    if (!this.landmarker) return { leftEye: null, rightEye: null, eyesClosed: false };

    const result = this.landmarker.detectForVideo(video, timestampMs);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      return { leftEye: null, rightEye: null, eyesClosed: false };
    }

    const face = result.faceLandmarks[0];
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

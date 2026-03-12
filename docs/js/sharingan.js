export class SharinganEffect {
  constructor(w, h) {
    this.width = w;
    this.height = h;
    this.rotationAngle = 0;
    this.activationAlpha = 0;
    this.image = null;
    this._loaded = false;
  }

  async loadImage(src = "assets/sharingan.png") {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.image = img;
        this._loaded = true;
        resolve();
      };
      img.onerror = () => {
        console.warn("Sharingan asset not found, will generate procedurally");
        this._generateProcedural();
        resolve();
      };
      img.src = src;
    });
  }

  _generateProcedural() {
    const size = 256;
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d");
    const center = size / 2;
    const radius = size / 2 - 4;

    // Red fill
    ctx.beginPath();
    ctx.arc(center, center, radius - 2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(180, 0, 0, 0.78)";
    ctx.fill();

    // Outer ring
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "rgb(200, 0, 0)";
    ctx.lineWidth = 3;
    ctx.stroke();

    // Black pupil
    ctx.beginPath();
    ctx.arc(center, center, radius / 4, 0, Math.PI * 2);
    ctx.fillStyle = "#000";
    ctx.fill();

    // Three tomoe
    for (let i = 0; i < 3; i++) {
      const angle = (i * 2 * Math.PI) / 3;
      const tomoeR = radius * 0.55;
      const tx = center + tomoeR * Math.cos(angle);
      const ty = center + tomoeR * Math.sin(angle);
      ctx.beginPath();
      ctx.arc(tx, ty, radius / 7, 0, Math.PI * 2);
      ctx.fillStyle = "#000";
      ctx.fill();
      const tailAngle = angle + 0.6;
      const tailX = center + tomoeR * 0.85 * Math.cos(tailAngle);
      const tailY = center + tomoeR * 0.85 * Math.sin(tailAngle);
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tailX, tailY);
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Red border
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "rgb(255, 0, 0)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Convert canvas to image
    const img = new Image();
    img.src = c.toDataURL();
    this.image = img;
    this._loaded = true;
  }

  render(ctx, eyeData, eyesClosed) {
    this.rotationAngle = (this.rotationAngle + 2) % 360;

    if (eyesClosed) {
      this.activationAlpha = Math.max(this.activationAlpha - 0.10, 0);
    } else {
      this.activationAlpha = Math.min(this.activationAlpha + 0.05, 1);
    }

    if (this.activationAlpha === 0 || !this._loaded) return;

    const { leftEye, rightEye } = eyeData;

    const drawEye = (eye) => {
      if (!eye) return;
      const [cx, cy] = eye.center;
      const size = Math.max(4, Math.floor(eye.radius * 0.9));
      const half = size / 2;

      ctx.save();
      ctx.globalAlpha = Math.min(this.activationAlpha, 1);
      ctx.translate(cx, cy);
      ctx.rotate((this.rotationAngle * Math.PI) / 180);
      ctx.drawImage(this.image, -half, -half, size, size);
      ctx.restore();
    };

    drawEye(leftEye);
    drawEye(rightEye);

    // Subtle red tint overlay
    if (this.activationAlpha > 0.3) {
      ctx.save();
      ctx.globalAlpha = 0.06 * this.activationAlpha;
      ctx.fillStyle = "#ff0000";
      ctx.fillRect(0, 0, this.width, this.height);
      ctx.restore();
    }
  }

  reset() {
    this.rotationAngle = 0;
    this.activationAlpha = 0;
  }
}

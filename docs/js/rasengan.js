export class RasenganEffect {
  static START_RADIUS = 75;
  static MAX_RADIUS = 150;
  static GROW_RATE = 0.4;
  static SHRINK_RATE = 2.0;

  constructor(w, h) {
    this.width = w;
    this.height = h;
    this.particles = [];
    this.ringAngle = 0;
    this.corePulse = 0;
    this.activationTime = 0;
    this.handPos = [w / 2, h / 2];
    this.baseRadius = RasenganEffect.START_RADIUS;
    this.fading = false;
    this.alive = false;
  }

  update(handPos = null) {
    if (handPos) this.handPos = handPos;

    if (this.fading) {
      const shrink = Math.max(RasenganEffect.SHRINK_RATE, this.baseRadius * 0.08);
      this.baseRadius -= shrink;
      if (this.baseRadius <= 0) {
        this.baseRadius = 0;
        this.alive = false;
        return;
      }
    } else {
      this.baseRadius = Math.min(this.baseRadius + RasenganEffect.GROW_RATE, RasenganEffect.MAX_RADIUS);
    }

    this.activationTime++;
    this.ringAngle += 8;
    this.corePulse = Math.sin(this.activationTime * 0.15) * (this.baseRadius * 0.06);

    // Spawn spiral particles
    const count = Math.max(2, Math.floor(6 * (this.baseRadius / RasenganEffect.MAX_RADIUS)));
    for (let i = 0; i < count; i++) {
      this.particles.push({
        angle: Math.random() * Math.PI * 2,
        dist: Math.random() * this.baseRadius * 1.2,
        speed: 0.5 + Math.random() * 1.5,
        orbitSpeed: 3 + Math.random() * 5,
        life: 10 + Math.floor(Math.random() * 20),
        size: Math.max(1, Math.floor(Math.random() * (this.baseRadius / 25 + 1)) + 1),
        brightness: 150 + Math.floor(Math.random() * 105),
      });
    }

    // Update particles
    for (const p of this.particles) {
      p.angle += (p.orbitSpeed * Math.PI) / 180;
      p.dist += p.speed * 0.2;
      p.life--;
    }
    this.particles = this.particles.filter(p => p.life > 0);
  }

  startFading() {
    this.fading = true;
  }

  render(ctx) {
    if (this.baseRadius <= 1) return;

    const [cx, cy] = this.handPos;
    const pulseR = Math.max(2, this.baseRadius + this.corePulse);

    // Outer glow
    for (let r = Math.floor(pulseR * 2); r > pulseR; r -= 3) {
      const alpha = Math.max(0, 0.12 * (1 - (r - pulseR) / pulseR));
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(50, 120, 255, ${alpha})`;
      ctx.fill();
    }

    // Spinning rings
    for (let i = 0; i < 3; i++) {
      const ringOffset = this.ringAngle + i * 120;
      const ringR = pulseR * 0.8;
      ctx.beginPath();
      for (let a = 0; a < 360; a += 5) {
        const rad = ((a + ringOffset) * Math.PI) / 180;
        const wobble = Math.sin(rad * 3 + this.activationTime * 0.1) * 5;
        const px = cx + (ringR + wobble) * Math.cos(rad);
        const py = cy + (ringR + wobble) * Math.sin(rad) * 0.4;
        if (a === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      const b = 180 + i * 25;
      ctx.strokeStyle = `rgb(${b >> 1}, ${b}, 255)`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Spiral particles
    for (const p of this.particles) {
      const px = cx + p.dist * Math.cos(p.angle);
      const py = cy + p.dist * Math.sin(p.angle);
      const lifeRatio = p.life / 40;
      const b = p.brightness;
      ctx.beginPath();
      ctx.arc(px, py, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${Math.floor(b * 0.3 * lifeRatio)}, ${Math.floor(b * 0.7 * lifeRatio)}, ${Math.floor(b * lifeRatio)})`;
      ctx.fill();
    }

    // Core sphere
    const coreR = Math.max(2, Math.floor(pulseR * 0.4));
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
    gradient.addColorStop(0, "rgba(220, 240, 255, 1)");
    gradient.addColorStop(0.5, "rgba(180, 220, 255, 0.8)");
    gradient.addColorStop(1, "rgba(100, 160, 255, 0)");
    ctx.beginPath();
    ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();
  }

  reset() {
    this.particles = [];
    this.ringAngle = 0;
    this.corePulse = 0;
    this.activationTime = 0;
    this.baseRadius = RasenganEffect.START_RADIUS;
    this.fading = false;
    this.alive = false;
  }
}

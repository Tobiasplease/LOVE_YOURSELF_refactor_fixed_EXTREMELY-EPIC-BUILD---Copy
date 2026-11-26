export class SpreadMeter {
  constructor(curtain, lines, radius = 0.15) {
    this.curtain = curtain;
    this.rest = curtain.pos.slice(0, curtain.clothCount * 3);
    const { nx, ny } = curtain;
    const rc = Math.max(1, Math.round(radius / curtain.dx));
    const rr = Math.max(1, Math.round(radius / curtain.dy));
    const offsets = [[-rc, 0], [rc, 0], [0, -rr], [0, rr], [-rc, -rr], [rc, -rr], [-rc, rr], [rc, rr]];
    this.points = lines.flatMap((l) => l.bridle.attach).map((k) => {
      const col = k % nx;
      const row = Math.floor(k / nx);
      const ring = offsets
        .map(([dc, dr]) => [col + dc, row + dr])
        .filter(([c, r]) => c >= 0 && c < nx && r >= 0 && r < ny && (c !== col || r !== row))
        .map(([c, r]) => r * nx + c);
      return { k, ring };
    });
    this.smoothed = null;
    this.sum = 0;
    this.samples = 0;
  }

  moved(k) {
    const p = this.curtain.pos;
    const q = this.rest;
    const i = k * 3;
    return Math.hypot(p[i] - q[i], p[i + 1] - q[i + 1], p[i + 2] - q[i + 2]);
  }

  measure() {
    let tent = 0;
    let active = 0;
    for (const { k, ring } of this.points) {
      const d = this.moved(k);
      if (d < 0.04) continue;
      const around = ring.reduce((sum, n) => sum + this.moved(n), 0) / ring.length;
      tent += Math.max(0, d - around);
      active++;
    }
    if (!active) return this.smoothed;
    const now = tent / active;
    this.sum += now;
    this.samples++;
    this.smoothed = this.smoothed === null ? now : this.smoothed * 0.95 + now * 0.05;
    return this.smoothed;
  }

  get overall() {
    return this.samples ? this.sum / this.samples : null;
  }
}

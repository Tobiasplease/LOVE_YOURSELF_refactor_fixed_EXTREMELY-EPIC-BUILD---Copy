import { gustEnvelope } from './wind.js';

export class CueEngine {
  constructor(gusts) {
    this.gusts = gusts;
    this.masterLog = [];
  }

  reset() {
    this.masterLog = [];
  }

  logMaster(cueT, value) {
    this.masterLog.push([cueT, value]);
    while (this.masterLog.length > 2 && this.masterLog[0][0] < cueT - 20) this.masterLog.shift();
  }

  masterAt(t) {
    const log = this.masterLog;
    if (!log.length) return 0;
    for (let i = log.length - 1; i >= 0; i--) if (log[i][0] <= t) return log[i][1];
    return log[0][1];
  }

  waveVector(p) {
    const a = (p.direction * Math.PI) / 180;
    return [Math.cos(a), Math.sin(a)];
  }

  delays(lines, p, railWidth, height) {
    const [dx, dy] = this.waveVector(p);
    const corners = [0, railWidth].flatMap((x) => [0, height].map((y) => x * dx + y * dy));
    const origin = Math.min(...corners);
    return lines.map((l) => {
      const lag = l.kind === 'hem' ? p.hemLag : 0;
      return (l.along * railWidth * dx + l.below * dy - origin) / p.waveSpeed + lag;
    });
  }

  shape(p, t) {
    let v;
    if (p.mode === 'gusts') v = gustEnvelope(this.gusts, t);
    else if (p.mode === 'swell') v = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / p.period);
    else v = this.masterAt(t);
    return p.invert ? 1 - v : v;
  }

  apply(lines, p, cueT, railWidth, height) {
    const delays = this.delays(lines, p, railWidth, height);
    const half = (p.shaper ?? 0) / 2;
    lines.forEach((l, i) => {
      const t = cueT - delays[i];
      const v = half > 0 ? 0.5 * this.shape(p, t) + 0.5 * this.shape(p, t - half) : this.shape(p, t);
      l.motor.target = (p.preload ?? 0) + p[`${l.kind}Pull`] * v;
    });
  }
}

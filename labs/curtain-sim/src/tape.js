const SCALE = 4000;

export class Tape {
  constructor(seconds = 20, rate = 30) {
    this.rate = rate;
    this.capacity = seconds * rate;
    this.frames = [];
  }

  clear() {
    this.frames = [];
  }

  get length() {
    return this.frames.length;
  }

  record(world) {
    const frame = {
      t: world.t,
      cueT: world.cueT,
      motors: Float32Array.from(world.lines.flatMap((l) => [l.motor.pos, l.motor.vel])),
      curtains: world.curtains.map((c) => {
        const q = new Int16Array(c.count * 3);
        for (let i = 0; i < q.length; i++) q[i] = Math.round(c.pos[i] * SCALE);
        return q;
      }),
    };
    this.frames.push(frame);
    if (this.frames.length > this.capacity) this.frames.shift();
  }

  show(index, world) {
    const frame = this.frames[index];
    if (!frame) return;
    world.curtains.forEach((c, n) => {
      const q = frame.curtains[n];
      if (!q) return;
      for (let i = 0; i < q.length; i++) c.pos[i] = q[i] / SCALE;
    });
    world.lines.forEach((l, i) => {
      l.motor.pos = frame.motors[i * 2];
      l.motor.vel = frame.motors[i * 2 + 1];
    });
    world.t = frame.t;
    world.cueT = frame.cueT;
  }

  resume(index, world) {
    this.show(index, world);
    const before = this.frames[Math.max(0, index - 1)];
    const now = this.frames[index];
    const span = Math.max(1e-6, now.t - before.t);
    world.curtains.forEach((c, n) => {
      const a = before.curtains[n];
      const b = now.curtains[n];
      if (!a || !b) return;
      for (let i = 0; i < b.length; i++) {
        c.vel[i] = index > 0 ? (b[i] - a[i]) / SCALE / span : 0;
        c.prev[i] = c.pos[i];
      }
    });
    this.frames.length = index + 1;
  }
}

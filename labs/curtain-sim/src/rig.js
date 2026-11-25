export class Motor {
  constructor({ maxSpeed = 0.8, maxAccel = 3 } = {}) {
    this.maxSpeed = maxSpeed;
    this.maxAccel = maxAccel;
    this.target = 0;
    this.pos = 0;
    this.vel = 0;
  }

  update(dt) {
    const error = this.target - this.pos;
    const brakingSpeed = Math.sqrt(2 * this.maxAccel * Math.abs(error));
    const wanted = Math.sign(error) * Math.min(this.maxSpeed, brakingSpeed);
    const maxChange = this.maxAccel * dt;
    this.vel += Math.max(-maxChange, Math.min(maxChange, wanted - this.vel));
    this.pos += this.vel * dt;
  }
}

const dist = (p, i, q) => Math.hypot(p[i] - q[0], p[i + 1] - q[1], p[i + 2] - q[2]);

export class Bridle {
  constructor(curtain, { attach, anchor, motor, elasticK = 40, reach = 0.35, slack = 0.03, pulleyMass = 0.004, legCompliance = 2e-5 }) {
    this.curtain = curtain;
    this.attach = attach;
    this.anchor = anchor.slice();
    this.motor = motor;
    this.elasticK = elasticK;
    this.slack = slack;
    this.legCompliance = legCompliance;

    const p = curtain.pos;
    const [a, b] = attach.map((k) => k * 3);
    const mid = [0, 1, 2].map((d) => (p[a + d] + p[b + d]) / 2);
    const toAnchor = [0, 1, 2].map((d) => this.anchor[d] - mid[d]);
    const len = Math.hypot(...toAnchor);
    const start = mid.map((m, d) => m + (toAnchor[d] / len) * reach);

    this.pulley = curtain.addParticle(start[0], start[1], start[2], pulleyMass, 8);
    this.restPulley = start;
    this.legLength = dist(p, a, start) + dist(p, b, start) + 0.005;
    this.updateRest();
    curtain.extras.push(this);
  }

  setAnchor(anchor) {
    this.anchor = anchor.slice();
    this.updateRest();
  }

  updateRest() {
    this.rest0 = Math.hypot(...this.restPulley.map((v, d) => v - this.anchor[d])) + this.slack;
  }

  get lineRest() {
    return this.rest0 - this.motor.pos;
  }

  get tension() {
    const stretch = dist(this.curtain.pos, this.pulley * 3, this.anchor) - this.lineRest;
    return Math.max(0, stretch) * this.elasticK;
  }

  solve(h) {
    const { pos, invMass } = this.curtain;
    const ip = this.pulley * 3;
    const wp = invMass[this.pulley];

    const ox = pos[ip] - this.anchor[0];
    const oy = pos[ip + 1] - this.anchor[1];
    const oz = pos[ip + 2] - this.anchor[2];
    const d = Math.hypot(ox, oy, oz);
    const over = d - this.lineRest;
    if (over > 0 && d > 1e-9) {
      const lambda = -over / (wp + 1 / this.elasticK / (h * h));
      const s = (wp * lambda) / d;
      pos[ip] += s * ox; pos[ip + 1] += s * oy; pos[ip + 2] += s * oz;
    }

    const ia = this.attach[0] * 3;
    const ib = this.attach[1] * 3;
    const wa = invMass[this.attach[0]];
    const wb = invMass[this.attach[1]];
    const ax = pos[ia] - pos[ip], ay = pos[ia + 1] - pos[ip + 1], az = pos[ia + 2] - pos[ip + 2];
    const bx = pos[ib] - pos[ip], by = pos[ib + 1] - pos[ip + 1], bz = pos[ib + 2] - pos[ip + 2];
    const da = Math.hypot(ax, ay, az);
    const db = Math.hypot(bx, by, bz);
    const legOver = da + db - this.legLength;
    if (legOver <= 0 || da < 1e-9 || db < 1e-9) return;

    const uax = ax / da, uay = ay / da, uaz = az / da;
    const ubx = bx / db, uby = by / db, ubz = bz / db;
    const gx = uax + ubx, gy = uay + uby, gz = uaz + ubz;
    const denom = wa + wb + wp * (gx * gx + gy * gy + gz * gz) + this.legCompliance / (h * h);
    const lambda = -legOver / denom;

    pos[ia] += wa * lambda * uax; pos[ia + 1] += wa * lambda * uay; pos[ia + 2] += wa * lambda * uaz;
    pos[ib] += wb * lambda * ubx; pos[ib + 1] += wb * lambda * uby; pos[ib + 2] += wb * lambda * ubz;
    pos[ip] -= wp * lambda * gx; pos[ip + 1] -= wp * lambda * gy; pos[ip + 2] -= wp * lambda * gz;
  }
}

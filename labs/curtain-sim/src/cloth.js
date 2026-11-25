const GRAVITY = -9.81;
const AIR_DENSITY = 1.2;

export const CLOTH_DEFAULTS = {
  railWidth: 1.4,
  fullness: 1.5,
  pleatPitch: 0.14,
  height: 2.4,
  spacing: 0.04,
  fabricGsm: 50,
  hemChainKgPerM: 0.06,
  railY: 2.55,
  x: 0,
  z: 0.3,
  wallZ: 0,
  stretchCompliance: 1e-7,
  shearCompliance: 1e-2,
  bendCompliance: 0.3,
  dragNormal: 1.1,
  dragTangent: 0.04,
  maxSpeed: 15,
};

function pleatShape(pitch, fullness) {
  const steps = 400;
  const build = (amp) => {
    const arc = [0];
    const xs = [0];
    const zs = [0];
    for (let k = 1; k <= steps; k++) {
      const x = (k / steps) * pitch;
      const z = amp * Math.sin((2 * Math.PI * x) / pitch);
      arc.push(arc[k - 1] + Math.hypot(x - xs[k - 1], z - zs[k - 1]));
      xs.push(x);
      zs.push(z);
    }
    return { arc, xs, zs };
  };
  let lo = 0;
  let hi = pitch;
  for (let it = 0; it < 40; it++) {
    const mid = (lo + hi) / 2;
    if (build(mid).arc[steps] < pitch * fullness) lo = mid;
    else hi = mid;
  }
  const { arc, xs, zs } = build(lo);
  const perPitch = arc[steps];

  return (u) => {
    const repeat = Math.floor(u / perPitch);
    const s = u - repeat * perPitch;
    let k = 1;
    while (k < steps && arc[k] < s) k++;
    const f = (s - arc[k - 1]) / (arc[k] - arc[k - 1] || 1);
    return [repeat * pitch + xs[k - 1] + f * (xs[k] - xs[k - 1]), zs[k - 1] + f * (zs[k] - zs[k - 1])];
  };
}

export class Curtain {
  constructor(options = {}) {
    const o = (this.o = { ...CLOTH_DEFAULTS, ...options });
    this.fabricWidth = o.railWidth * o.fullness;
    this.nx = Math.round(this.fabricWidth / o.spacing) + 1;
    this.ny = Math.round(o.height / o.spacing) + 1;
    this.dx = this.fabricWidth / (this.nx - 1);
    this.dy = o.height / (this.ny - 1);
    this.clothCount = this.nx * this.ny;

    const capacity = this.clothCount + 64;
    this.pos = new Float32Array(capacity * 3);
    this.prev = new Float32Array(capacity * 3);
    this.vel = new Float32Array(capacity * 3);
    this.force = new Float32Array(capacity * 3);
    this.wind = new Float32Array(capacity * 3);
    this.invMass = new Float32Array(capacity);
    this.damping = new Float32Array(capacity);
    this.count = 0;

    this.pending = { a: [], b: [], rest: [], comp: [] };
    this.dirty = true;
    this.extras = [];

    this.buildFabric();
    this.buildTriangles();
  }

  index(col, row) {
    return row * this.nx + col;
  }

  addParticle(x, y, z, mass, damping = 0) {
    const k = this.count++;
    this.pos.set([x, y, z], k * 3);
    this.prev.set([x, y, z], k * 3);
    this.invMass[k] = mass > 0 ? 1 / mass : 0;
    this.damping[k] = damping;
    return k;
  }

  distance(a, b) {
    const p = this.pos;
    return Math.hypot(p[a * 3] - p[b * 3], p[a * 3 + 1] - p[b * 3 + 1], p[a * 3 + 2] - p[b * 3 + 2]);
  }

  addLink(a, b, compliance, rest = this.distance(a, b)) {
    this.pending.a.push(a);
    this.pending.b.push(b);
    this.pending.rest.push(rest);
    this.pending.comp.push(compliance);
    this.dirty = true;
  }

  buildFabric() {
    const o = this.o;
    const { nx, ny, dx, dy } = this;
    const pleat = pleatShape(o.pleatPitch, o.fullness);
    const railStart = o.x - (pleat(this.fabricWidth)[0]) / 2;
    const particleMass = ((o.fabricGsm / 1000) * this.fabricWidth * o.height) / this.clothCount;
    const chainMassPerParticle = (o.hemChainKgPerM * this.fabricWidth) / nx;

    for (let row = 0; row < ny; row++) {
      for (let col = 0; col < nx; col++) {
        const [px, pz] = pleat(col * dx);
        const x = railStart + px;
        const z = o.z + pz;
        const y = o.railY - row * dy;
        const isHem = row === ny - 1;
        const mass = row === 0 ? 0 : particleMass + (isHem ? chainMassPerParticle : 0);
        this.addParticle(x, y, z, mass);
      }
    }

    for (let row = 0; row < ny; row++) {
      for (let col = 0; col < nx; col++) {
        const k = this.index(col, row);
        if (col + 1 < nx) this.addLink(k, k + 1, o.stretchCompliance);
        if (row + 1 < ny) this.addLink(k, k + nx, o.stretchCompliance);
        if (col + 1 < nx && row + 1 < ny) {
          this.addLink(k, k + nx + 1, o.shearCompliance);
          this.addLink(k + 1, k + nx, o.shearCompliance);
        }
        if (col + 2 < nx) this.addLink(k, k + 2, o.bendCompliance);
        if (row + 2 < ny) this.addLink(k, k + 2 * nx, o.bendCompliance);
      }
    }
  }

  addRod(row, centerCol, lengthM, rodMassKg, compliance = 2e-6) {
    const half = Math.max(1, Math.round(lengthM / 2 / this.dx));
    const first = Math.max(0, centerCol - half);
    const last = Math.min(this.nx - 1, centerCol + half);
    const particles = [];
    for (let col = first; col <= last; col++) particles.push(this.index(col, row));

    const extraMass = rodMassKg / particles.length;
    for (const k of particles) this.invMass[k] = 1 / (1 / this.invMass[k] + extraMass);

    for (let i = 0; i < particles.length; i++) {
      for (let span = 2; i + span < particles.length; span++) {
        this.addLink(particles[i], particles[i + span], compliance, span * this.dx);
      }
    }
    return particles;
  }

  buildTriangles() {
    const { nx, ny } = this;
    const tris = [];
    for (let row = 0; row + 1 < ny; row++) {
      for (let col = 0; col + 1 < nx; col++) {
        const a = this.index(col, row);
        const b = a + 1;
        const c = a + nx;
        const d = c + 1;
        tris.push(a, c, b, b, c, d);
      }
    }
    this.triangles = new Uint32Array(tris);
  }

  pack() {
    const p = this.pending;
    this.linkA = Int32Array.from(p.a);
    this.linkB = Int32Array.from(p.b);
    this.linkRest = Float32Array.from(p.rest);
    this.linkComp = Float32Array.from(p.comp);
    this.dirty = false;
  }

  computeAirForces() {
    const { pos, vel, wind, force, triangles, o } = this;
    force.fill(0);
    const kn = 0.5 * AIR_DENSITY * o.dragNormal;
    const kt = 0.5 * AIR_DENSITY * o.dragTangent;

    for (let t = 0; t < triangles.length; t += 3) {
      const a = triangles[t] * 3;
      const b = triangles[t + 1] * 3;
      const c = triangles[t + 2] * 3;

      const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
      const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
      let nx = e1y * e2z - e1z * e2y;
      let ny = e1z * e2x - e1x * e2z;
      let nz = e1x * e2y - e1y * e2x;
      const twiceArea = Math.hypot(nx, ny, nz);
      if (twiceArea < 1e-12) continue;
      nx /= twiceArea;
      ny /= twiceArea;
      nz /= twiceArea;
      const area = twiceArea / 2;

      const rx = (vel[a] + vel[b] + vel[c] - wind[a] - wind[b] - wind[c]) / 3;
      const ry = (vel[a + 1] + vel[b + 1] + vel[c + 1] - wind[a + 1] - wind[b + 1] - wind[c + 1]) / 3;
      const rz = (vel[a + 2] + vel[b + 2] + vel[c + 2] - wind[a + 2] - wind[b + 2] - wind[c + 2]) / 3;

      const vn = rx * nx + ry * ny + rz * nz;
      const tx = rx - vn * nx;
      const ty = ry - vn * ny;
      const tz = rz - vn * nz;
      const vt = Math.hypot(tx, ty, tz);

      const pushN = (-kn * area * vn * Math.abs(vn)) / 3;
      const pushT = (-kt * area * vt) / 3;
      const fx = pushN * nx + pushT * tx;
      const fy = pushN * ny + pushT * ty;
      const fz = pushN * nz + pushT * tz;

      force[a] += fx; force[a + 1] += fy; force[a + 2] += fz;
      force[b] += fx; force[b + 1] += fy; force[b + 2] += fz;
      force[c] += fx; force[c + 1] += fy; force[c + 2] += fz;
    }
  }

  solveLinks(h) {
    const { pos, invMass, linkA, linkB, linkRest, linkComp } = this;
    const h2 = h * h;
    for (let n = 0; n < linkA.length; n++) {
      const a = linkA[n];
      const b = linkB[n];
      const wa = invMass[a];
      const wb = invMass[b];
      const w = wa + wb;
      if (w === 0) continue;
      const ia = a * 3;
      const ib = b * 3;
      const dx = pos[ia] - pos[ib];
      const dy = pos[ia + 1] - pos[ib + 1];
      const dz = pos[ia + 2] - pos[ib + 2];
      const d = Math.hypot(dx, dy, dz);
      if (d < 1e-9) continue;
      const lambda = -(d - linkRest[n]) / (w + linkComp[n] / h2);
      const s = lambda / d;
      pos[ia] += wa * s * dx; pos[ia + 1] += wa * s * dy; pos[ia + 2] += wa * s * dz;
      pos[ib] -= wb * s * dx; pos[ib + 1] -= wb * s * dy; pos[ib + 2] -= wb * s * dz;
    }
  }

  collide() {
    const { pos, o } = this;
    const wallLimit = o.wallZ + 0.01;
    for (let k = 0; k < this.clothCount; k++) {
      const i = k * 3;
      if (pos[i + 2] < wallLimit) pos[i + 2] = wallLimit;
      if (pos[i + 1] < 0.003) pos[i + 1] = 0.003;
    }
  }

  step(dt, substeps) {
    if (this.dirty) this.pack();
    const h = dt / substeps;
    const { pos, prev, vel, force, invMass, damping } = this;

    for (let s = 0; s < substeps; s++) {
      this.computeAirForces();

      for (let k = 0; k < this.count; k++) {
        const w = invMass[k];
        if (w === 0) continue;
        const i = k * 3;
        const keep = 1 - damping[k] * h;
        vel[i] = (vel[i] + force[i] * w * h) * keep;
        vel[i + 1] = (vel[i + 1] + (GRAVITY + force[i + 1] * w) * h) * keep;
        vel[i + 2] = (vel[i + 2] + force[i + 2] * w * h) * keep;
        prev[i] = pos[i]; prev[i + 1] = pos[i + 1]; prev[i + 2] = pos[i + 2];
        pos[i] += vel[i] * h; pos[i + 1] += vel[i + 1] * h; pos[i + 2] += vel[i + 2] * h;
      }

      this.solveLinks(h);
      for (const extra of this.extras) extra.solve(h);
      this.collide();

      const maxSpeed = this.o.maxSpeed;
      for (let k = 0; k < this.count; k++) {
        if (invMass[k] === 0) continue;
        const i = k * 3;
        vel[i] = (pos[i] - prev[i]) / h;
        vel[i + 1] = (pos[i + 1] - prev[i + 1]) / h;
        vel[i + 2] = (pos[i + 2] - prev[i + 2]) / h;
        const speed = Math.hypot(vel[i], vel[i + 1], vel[i + 2]);
        if (speed > maxSpeed) {
          const f = maxSpeed / speed;
          vel[i] *= f; vel[i + 1] *= f; vel[i + 2] *= f;
        }
      }
    }
  }
}

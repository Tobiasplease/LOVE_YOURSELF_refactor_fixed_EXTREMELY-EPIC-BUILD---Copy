function seededRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeGusts(seed = 7, duration = 1200) {
  const rand = seededRandom(seed);
  const gusts = [];
  let t = 1.5;
  while (t < duration) {
    const gust = {
      start: t,
      attack: 0.4 + rand() * 0.8,
      hold: 0.5 + rand() * 2.5,
      release: 1.8 + rand() * 3.0,
      peak: 0.45 + rand() * 0.55,
    };
    gusts.push(gust);
    t += gust.attack + gust.hold + gust.release * 0.6 + 2 + rand() * 9;
  }
  return gusts;
}

const smooth = (x) => x * x * (3 - 2 * x);

export function gustEnvelope(gusts, t) {
  let value = 0;
  for (const g of gusts) {
    const local = t - g.start;
    if (local < 0) break;
    const releaseStart = g.attack + g.hold;
    if (local > releaseStart + g.release * 2) continue;
    let e;
    if (local < g.attack) e = smooth(local / g.attack);
    else if (local < releaseStart) e = 1;
    else e = Math.exp((-3 * (local - releaseStart)) / g.release);
    value = Math.max(value, e * g.peak);
  }
  return value;
}

function hash(i, j, k) {
  let h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(k, 2147483647);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h & 0xffff) / 32768 - 1;
}

export function noise3(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const fx = smooth(x - xi), fy = smooth(y - yi), fz = smooth(z - zi);
  const lerp = (a, b, t) => a + (b - a) * t;
  const corner = (dx, dy, dz) => hash(xi + dx, yi + dy, zi + dz);
  return lerp(
    lerp(lerp(corner(0, 0, 0), corner(1, 0, 0), fx), lerp(corner(0, 1, 0), corner(1, 1, 0), fx), fy),
    lerp(lerp(corner(0, 0, 1), corner(1, 0, 1), fx), lerp(corner(0, 1, 1), corner(1, 1, 1), fx), fy),
    fz,
  );
}

export class WindField {
  constructor({ gusts, xRef, yTop, base = 0.4, gustSpeed = 3.0, turbulence = 0.45, frontSpeedX = 2.2, frontSpeedY = 4.0 }) {
    Object.assign(this, { gusts, xRef, yTop, base, gustSpeed, turbulence, frontSpeedX, frontSpeedY });
    this.strength = 1;
  }

  arrivalDelay(x, y) {
    return (x - this.xRef) / this.frontSpeedX + (this.yTop - y) / this.frontSpeedY;
  }

  fill(curtain, t) {
    const { pos, wind } = curtain;
    for (let k = 0; k < curtain.clothCount; k++) {
      const i = k * 3;
      const x = pos[i], y = pos[i + 1];
      const g = gustEnvelope(this.gusts, t - this.arrivalDelay(x, y));
      const slow = noise3(x * 1.3 - t * 1.6, y * 1.1 + t * 0.3, t * 0.4);
      const fast = noise3(x * 3.1 - t * 3.0 + 17, y * 2.7, t * 1.3 + 5);
      const speed = (this.base + this.gustSpeed * g) * this.strength * (1 + this.turbulence * (0.7 * slow + 0.5 * fast));
      wind[i] = speed * (0.35 + 0.25 * fast);
      wind[i + 1] = speed * 0.15 * slow;
      wind[i + 2] = speed;
    }
  }
}

export function createPanel(root, params, spec, onChange) {
  for (const item of spec) {
    if (item.heading) {
      const h = document.createElement('h3');
      h.textContent = item.heading;
      root.appendChild(h);
      continue;
    }
    const row = document.createElement('label');
    row.className = item.type === 'check' ? 'row check' : 'row';
    const name = document.createElement('span');
    name.textContent = item.label;
    const input = document.createElement('input');

    if (item.type === 'check') {
      input.type = 'checkbox';
      input.checked = params[item.key];
      input.addEventListener('change', () => {
        params[item.key] = input.checked;
        onChange(item);
      });
      row.append(input, name);
    } else {
      input.type = 'range';
      Object.assign(input, { min: item.min, max: item.max, step: item.step ?? 0.01, value: params[item.key] });
      const value = document.createElement('output');
      const show = () => (value.textContent = `${Number(params[item.key]).toFixed(item.digits ?? 2)} ${item.unit ?? ''}`);
      show();
      input.addEventListener('input', () => {
        params[item.key] = Number(input.value);
        show();
        onChange(item);
      });
      row.append(name, value, input);
    }
    root.appendChild(row);
  }
}

export class Scope {
  constructor(canvas, seconds = 12, traces = []) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.seconds = seconds;
    this.traces = traces;
    this.samples = [];
  }

  push(t, values) {
    this.samples.push([t, ...values]);
    while (this.samples.length && this.samples[0][0] < t - this.seconds) this.samples.shift();
  }

  clear() {
    this.samples = [];
  }

  draw() {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    ctx.fillRect(0, 0, w, h);
    if (this.samples.length < 2) return;
    const tEnd = this.samples[this.samples.length - 1][0];
    this.traces.forEach((trace, n) => {
      ctx.strokeStyle = trace.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      this.samples.forEach((s, i) => {
        const x = w - ((tEnd - s[0]) / this.seconds) * w;
        const y = h - 4 - Math.min(1, s[n + 1] / trace.max) * (h - 8);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }
}

// Usage: node debug/curtain_sim_snapshot.mjs <outDir> [times, e.g. 0,3,5,8 | video:start:end:fps] [json params override] [view] [phone]
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const root = resolve(new URL('../labs/curtain-sim', import.meta.url).pathname);
const outDir = resolve(process.argv[2] ?? '.');
const spec = process.argv[3] ?? '0,3,6';
const video = spec.startsWith('video:');
let times;
if (video) {
  const [start, end, fps] = spec.slice(6).split(':').map(Number);
  times = [];
  for (let t = start; t <= end; t += 1 / fps) times.push(Number(t.toFixed(4)));
} else times = spec.split(',').map(Number);
const view = process.argv[5] ?? 'angle';
const phone = process.argv[6] === 'phone';
const overrides = JSON.parse(process.argv[4] ?? '{}');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' };

const server = createServer(async (req, res) => {
  try {
    const path = join(root, decodeURIComponent(req.url.split('?')[0]).replace(/\/$/, '/index.html'));
    res.writeHead(200, { 'Content-Type': types[extname(path)] ?? 'application/octet-stream' });
    res.end(await readFile(path));
  } catch {
    res.writeHead(404).end();
  }
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const viewport = phone ? { width: 390, height: 844 } : video ? { width: 1280, height: 720 } : { width: 1400, height: 860 };
const page = await browser.newPage({ viewport, deviceScaleFactor: 1, isMobile: phone, hasTouch: phone });
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(`http://localhost:${port}/`);
await page.waitForFunction(() => window.curtainSim?.world);
await page.evaluate(
  ([o, v]) => {
    const sim = window.curtainSim;
    sim.setTransport('pause');
    Object.assign(sim.params, o);
    if (Object.keys(o).length) sim.build();
    sim.refreshAll();
    sim.setView(v);
  },
  [overrides, view],
);
if (video) await page.addStyleTag({ content: '#panel, #hud, #status { display: none !important; }' });
let frameNo = 0;

for (const target of times) {
  const ms = await page.evaluate((target) => {
    const sim = window.curtainSim;
    const start = performance.now();
    let steps = 0;
    while (sim.t < target) {
      sim.simulate();
      steps++;
    }
    return steps ? (performance.now() - start) / steps : 0;
  }, target);
  await page.waitForTimeout(video ? 40 : 150);
  const file = join(outDir, video ? `f${String(frameNo++).padStart(5, '0')}.png` : `t${String(target).padStart(5, '0')}.png`);
  await page.screenshot({ path: file });
  if (!video || frameNo % 30 === 0) console.log(`t=${target}s  ${ms.toFixed(1)} ms/frame  -> ${file}`);
}

await browser.close();
server.close();

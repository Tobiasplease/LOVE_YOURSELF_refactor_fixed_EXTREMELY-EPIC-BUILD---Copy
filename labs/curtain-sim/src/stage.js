import * as THREE from 'three';
import { RectAreaLightUniformsLib } from '../vendor/three/addons/RectAreaLightUniformsLib.js';

export const ROOM = { width: 7, depth: 7, ceiling: 2.6 };
export const LIGHTBOX = { width: 1.2, height: 2.0, centerY: 1.4 };

export function kelvinToRGB(kelvin) {
  const t = kelvin / 100;
  const r = t <= 66 ? 255 : 329.7 * Math.pow(t - 60, -0.1332);
  const g = t <= 66 ? 99.47 * Math.log(t) - 161.12 : 288.12 * Math.pow(t - 60, -0.0755);
  const b = t >= 66 ? 255 : t <= 19 ? 0 : 138.52 * Math.log(t - 10) - 305.04;
  const c = (v) => Math.min(255, Math.max(0, v)) / 255;
  return new THREE.Color(c(r), c(g), c(b)).convertSRGBToLinear();
}

export function createStage(canvas, windowXs) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  RectAreaLightUniformsLib.init();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 50);
  const orbit = { target: new THREE.Vector3(0, 1.25, 0.6), yaw: 0, pitch: 0, distance: 6 };
  const views = {
    front: { yaw: 0, pitch: 0.02, distance: 6.2 },
    angle: { yaw: 0.5, pitch: 0.06, distance: 6.0 },
    side: { yaw: 1.35, pitch: 0.05, distance: 5.0 },
  };
  const placeCamera = () => {
    const { target, yaw, pitch, distance } = orbit;
    camera.position.set(
      target.x + distance * Math.sin(yaw) * Math.cos(pitch),
      target.y + distance * Math.sin(pitch),
      target.z + distance * Math.cos(yaw) * Math.cos(pitch),
    );
    camera.lookAt(target);
  };
  const setView = (name) => {
    Object.assign(orbit, views[name]);
    placeCamera();
  };
  setView('angle');

  let drag = null;
  canvas.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    orbit.yaw -= (e.clientX - drag.x) * 0.005;
    orbit.pitch = Math.max(-0.2, Math.min(0.9, orbit.pitch + (e.clientY - drag.y) * 0.004));
    drag = { x: e.clientX, y: e.clientY };
    placeCamera();
  });
  canvas.addEventListener('pointerup', () => (drag = null));
  canvas.addEventListener('wheel', (e) => {
    orbit.distance = Math.max(1.5, Math.min(9, orbit.distance * (1 + e.deltaY * 0.001)));
    placeCamera();
  });

  const surface = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.95, metalness: 0 });
  const wallMat = surface(0x4a4642);
  const floorMat = surface(0x55504a);

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.width, ROOM.depth), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.z = ROOM.depth / 2;
  scene.add(floor);

  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.width, ROOM.depth), surface(0x3a3734));
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, ROOM.ceiling, ROOM.depth / 2);
  scene.add(ceiling);

  const back = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.width, ROOM.ceiling), wallMat);
  back.position.set(0, ROOM.ceiling / 2, 0);
  scene.add(back);

  for (const side of [-1, 1]) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.depth, ROOM.ceiling), wallMat);
    wall.rotation.y = -side * Math.PI / 2;
    wall.position.set((side * ROOM.width) / 2, ROOM.ceiling / 2, ROOM.depth / 2);
    scene.add(wall);
  }

  const lightboxes = windowXs.map((x) => {
    const glow = new THREE.Mesh(
      new THREE.PlaneGeometry(LIGHTBOX.width, LIGHTBOX.height),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    glow.position.set(x, LIGHTBOX.centerY, 0.004);
    scene.add(glow);

    const frameMat = surface(0x2a2724);
    const t = 0.05;
    const frameParts = [
      [LIGHTBOX.width + 2 * t, t, x, LIGHTBOX.centerY + LIGHTBOX.height / 2 + t / 2],
      [LIGHTBOX.width + 2 * t, t, x, LIGHTBOX.centerY - LIGHTBOX.height / 2 - t / 2],
      [t, LIGHTBOX.height, x - LIGHTBOX.width / 2 - t / 2, LIGHTBOX.centerY],
      [t, LIGHTBOX.height, x + LIGHTBOX.width / 2 + t / 2, LIGHTBOX.centerY],
    ];
    for (const [w, h, fx, fy] of frameParts) {
      const part = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.03), frameMat);
      part.position.set(fx, fy, 0.015);
      scene.add(part);
    }

    const light = new THREE.RectAreaLight(0xffffff, 6, LIGHTBOX.width, LIGHTBOX.height);
    light.position.set(x, LIGHTBOX.centerY, 0.02);
    light.lookAt(x, LIGHTBOX.centerY, 1);
    scene.add(light);

    const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 1.6, 8), surface(0x1c1a18));
    rail.rotation.z = Math.PI / 2;
    rail.position.set(x, 2.565, 0.3);
    scene.add(rail);

    return { glow, light };
  });

  const setKelvin = (kelvin, brightness = 2.4) => {
    const color = kelvinToRGB(kelvin);
    for (const box of lightboxes) {
      box.glow.material.color.copy(color).multiplyScalar(brightness);
      box.light.color.copy(color);
    }
    return color;
  };

  const resize = () => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener('resize', resize);
  resize();

  return { renderer, scene, camera, setKelvin, setView };
}

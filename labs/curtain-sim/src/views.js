import * as THREE from 'three';
import { LIGHTBOX } from './stage.js';

const voileVertex = /* glsl */ `
  varying vec3 vWorld;
  varying vec3 vNormal;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    vNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const voileFragment = /* glsl */ `
  uniform vec3 uLight;
  uniform vec4 uWindow;
  uniform float uSheerness;
  varying vec3 vWorld;
  varying vec3 vNormal;

  float band(float v, float lo, float hi, float soft) {
    return smoothstep(lo - soft, lo + soft, v) * (1.0 - smoothstep(hi - soft, hi + soft, v));
  }

  void main() {
    vec3 n = normalize(vNormal);
    vec3 toEye = normalize(cameraPosition - vWorld);
    float faceOn = abs(dot(n, toEye));
    float layers = 1.0 / max(faceOn, 0.12);
    float alpha = 1.0 - pow(1.0 - uSheerness, layers);

    float depth = max(vWorld.z, 0.0);
    float soft = 0.06 + 0.45 * depth;
    float behind = band(vWorld.x, uWindow.x, uWindow.y, soft) * band(vWorld.y, uWindow.z, uWindow.w, soft);
    float glow = mix(0.03, 1.0, behind) / (1.0 + 0.5 * depth * depth);
    float catchesLight = 0.35 + 0.65 * abs(n.z);

    gl_FragColor = vec4(uLight * glow * catchesLight * 0.9, alpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

export class CurtainView {
  constructor(scene, curtain, windowX) {
    this.curtain = curtain;
    const geometry = new THREE.BufferGeometry();
    this.positions = new THREE.BufferAttribute(curtain.pos.subarray(0, curtain.clothCount * 3), 3);
    this.positions.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', this.positions);
    geometry.setIndex(new THREE.BufferAttribute(curtain.triangles, 1));
    geometry.computeVertexNormals();
    this.geometry = geometry;

    this.material = new THREE.ShaderMaterial({
      vertexShader: voileVertex,
      fragmentShader: voileFragment,
      uniforms: {
        uLight: { value: new THREE.Color(1, 1, 1) },
        uWindow: {
          value: new THREE.Vector4(
            windowX - LIGHTBOX.width / 2,
            windowX + LIGHTBOX.width / 2,
            LIGHTBOX.centerY - LIGHTBOX.height / 2,
            LIGHTBOX.centerY + LIGHTBOX.height / 2,
          ),
        },
        uSheerness: { value: 0.32 },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }

  setLight(color, brightness) {
    this.material.uniforms.uLight.value.copy(color).multiplyScalar(brightness);
  }

  update() {
    this.positions.needsUpdate = true;
    this.geometry.computeVertexNormals();
  }

  dispose(scene) {
    scene.remove(this.mesh);
    this.geometry.dispose();
    this.material.dispose();
  }
}

const SLACK = [0.35, 0.36, 0.38];
const TAUT = [0.85, 0.88, 0.92];
const LIMIT = [0.95, 0.62, 0.2];
const ROD = [0.75, 0.7, 0.6];

export class RigView {
  constructor(scene, curtain, lines, rods) {
    this.curtain = curtain;
    this.lines = lines;
    this.rods = rods;
    this.group = new THREE.Group();
    scene.add(this.group);

    const segments = lines.length * 3 + rods.reduce((n, r) => n + r.length - 1, 0);
    this.linePositions = new Float32Array(segments * 6);
    this.lineColors = new Float32Array(segments * 6);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.linePositions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(this.lineColors, 3));
    this.segments = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6, toneMapped: false }),
    );
    this.segments.frustumCulled = false;
    this.group.add(this.segments);

    const metal = new THREE.MeshBasicMaterial({ color: 0x8d949b });
    const anchorGeo = new THREE.TorusGeometry(0.022, 0.005, 6, 16);
    const floatGeo = new THREE.SphereGeometry(0.008, 8, 6);
    this.anchors = lines.map(() => this.group.add(new THREE.Mesh(anchorGeo, metal)).children.at(-1));
    this.floats = lines.map(() => this.group.add(new THREE.Mesh(floatGeo, metal)).children.at(-1));
  }

  update() {
    const p = this.curtain.pos;
    const out = this.linePositions;
    const col = this.lineColors;
    let o = 0;
    const seg = (a, b, c) => {
      out.set(a, o);
      out.set(b, o + 3);
      col.set(c, o);
      col.set(c, o + 3);
      o += 6;
    };
    const at = (k) => [p[k * 3], p[k * 3 + 1], p[k * 3 + 2]];

    this.lines.forEach((line, n) => {
      const { bridle, motor } = line;
      const pulley = at(bridle.pulley);
      const colour = motor.saturated ? LIMIT : bridle.tension > 0.02 ? TAUT : SLACK;
      seg(pulley, bridle.anchor, colour);
      seg(pulley, at(bridle.attach[0]), colour);
      seg(pulley, at(bridle.attach[1]), colour);
      this.anchors[n].position.fromArray(bridle.anchor);
      this.floats[n].position.fromArray(pulley);
    });
    for (const rod of this.rods) {
      for (let i = 0; i + 1 < rod.length; i++) seg(at(rod[i]), at(rod[i + 1]), ROD);
    }
    this.segments.geometry.attributes.position.needsUpdate = true;
    this.segments.geometry.attributes.color.needsUpdate = true;
  }

  dispose(scene) {
    scene.remove(this.group);
    this.segments.geometry.dispose();
  }
}

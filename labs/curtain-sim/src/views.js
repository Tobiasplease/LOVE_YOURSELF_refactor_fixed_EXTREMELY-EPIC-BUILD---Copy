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

export class RigView {
  constructor(scene, curtain, bridles, rods) {
    this.curtain = curtain;
    this.bridles = bridles;
    this.rods = rods;
    this.group = new THREE.Group();
    scene.add(this.group);

    const lineCount = bridles.length * 3 + rods.reduce((n, r) => n + r.length - 1, 0);
    this.linePositions = new Float32Array(lineCount * 6);
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(this.linePositions, 3));
    this.lines = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({ color: 0xc8d0d8, transparent: true, opacity: 0.45 }),
    );
    this.lines.frustumCulled = false;
    this.group.add(this.lines);

    const pulleyMat = new THREE.MeshBasicMaterial({ color: 0x9aa3ab });
    this.anchorMeshes = bridles.map(() => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(0.025, 0.006, 8, 20), pulleyMat);
      this.group.add(m);
      return m;
    });
    this.floatMeshes = bridles.map(() => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.009, 10, 8), pulleyMat);
      this.group.add(m);
      return m;
    });
  }

  update() {
    const p = this.curtain.pos;
    const out = this.linePositions;
    let o = 0;
    const seg = (a, b) => {
      out.set(a, o);
      out.set(b, o + 3);
      o += 6;
    };
    const at = (k) => [p[k * 3], p[k * 3 + 1], p[k * 3 + 2]];

    this.bridles.forEach((bridle, n) => {
      const pulley = at(bridle.pulley);
      seg(pulley, bridle.anchor);
      seg(pulley, at(bridle.attach[0]));
      seg(pulley, at(bridle.attach[1]));
      this.anchorMeshes[n].position.fromArray(bridle.anchor);
      this.floatMeshes[n].position.fromArray(pulley);
    });
    for (const rod of this.rods) {
      for (let i = 0; i + 1 < rod.length; i++) seg(at(rod[i]), at(rod[i + 1]));
    }
    this.lines.geometry.attributes.position.needsUpdate = true;
  }

  dispose(scene) {
    scene.remove(this.group);
  }
}

/**
 * Cross-Border Liquidity Desk — Three.js visualisation.
 *
 * Renders the state of LiquidityDeskEnv as a physical treasury vault:
 * four glass columns of liquid inventory (NGN, RWF, KES and a central USDT
 * settlement rail), joined by a pipe manifold that pulses whenever the agent
 * rebalances. Liquid level IS the pool balance, so the agent's policy is
 * legible at a glance.
 *
 * State arrives over a websocket as one JSON snapshot per environment step
 * (see environment/rendering.py and LiquidityDeskEnv.render_state).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// --------------------------------------------------------------------------
// Error reporting
//
// A WebGL scene that fails at runtime shows a black canvas with the HUD still
// drawn on top, which is indistinguishable from "nothing is happening". These
// handlers surface the error on the page and relay it back over the websocket
// so it also appears in the terminal running the simulation.
// --------------------------------------------------------------------------

const errorLog = [];
let errorSink = null;

function reportError(label, err) {
  const message = err && err.message ? err.message : String(err);
  const stack = err && err.stack ? err.stack : '';
  const entry = { type: 'error', label, message, stack };

  errorLog.push(entry);
  console.error(`[scene] ${label}:`, err);

  const box = document.getElementById('crash');
  if (box) {
    box.style.display = 'block';
    box.innerHTML = errorLog
      .map((e) => `<b>${e.label}</b>\n${e.message}\n${e.stack}`)
      .join('\n\n');
  }

  if (errorSink) {
    try {
      errorSink(JSON.stringify(entry));
    } catch (_) { /* socket gone; the page still shows it */ }
  }
}

window.addEventListener('error', (e) => {
  reportError('uncaught', e.error || e.message);
});
window.addEventListener('unhandledrejection', (e) => {
  reportError('unhandled promise rejection', e.reason);
});

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------

const HUB = 3;

const VAULTS = [
  { name: 'NGN',  colour: 0x22c55e, pos: [-6.2, 0,  2.2], radius: 1.15, height: 6.5 },
  { name: 'RWF',  colour: 0x6366f1, pos: [ 0.0, 0,  4.2], radius: 1.15, height: 6.5 },
  { name: 'KES',  colour: 0xf59e0b, pos: [ 6.2, 0,  2.2], radius: 1.15, height: 6.5 },
  { name: 'USDT', colour: 0x06b6d4, pos: [ 0.0, 0, -2.2], radius: 1.70, height: 8.0 },
];

const CSS_COLOUR = ['#22c55e', '#6366f1', '#f59e0b', '#06b6d4'];
const SPARK_HISTORY = 240;
const FEED_LENGTH = 7;

// --------------------------------------------------------------------------
// Renderer, scene, camera
// --------------------------------------------------------------------------

const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);
scene.fog = new THREE.Fog(0x05070a, 26, 62);

const camera = new THREE.PerspectiveCamera(
  42, window.innerWidth / window.innerHeight, 0.1, 200,
);
camera.position.set(0.5, 7.5, 17.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 3.0, 0.5);
controls.enableDamping = true;
controls.dampingFactor = 0.055;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 8;
controls.maxDistance = 42;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.28;
controls.addEventListener('start', () => { controls.autoRotate = false; });

// Procedural environment map: gives the glass real reflections with no HDRI
// file to download, which keeps the whole scene usable offline.
{
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;
  pmrem.dispose();
}

// --------------------------------------------------------------------------
// Lighting
// --------------------------------------------------------------------------

scene.add(new THREE.AmbientLight(0x3a4a60, 0.55));

const keyLight = new THREE.DirectionalLight(0xdce8ff, 1.5);
keyLight.position.set(9, 16, 11);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 1;
keyLight.shadow.camera.far = 60;
keyLight.shadow.camera.left = -22;
keyLight.shadow.camera.right = 22;
keyLight.shadow.camera.top = 22;
keyLight.shadow.camera.bottom = -22;
keyLight.shadow.bias = -0.0012;
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0x2b6cff, 0.75);
rimLight.position.set(-12, 7, -14);
scene.add(rimLight);

scene.add(new THREE.HemisphereLight(0x8fb6ff, 0x0a0f16, 0.35));

// --------------------------------------------------------------------------
// Floor
// --------------------------------------------------------------------------

{
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(90, 90),
    new THREE.MeshStandardMaterial({
      color: 0x0d1319,
      roughness: 0.42,
      metalness: 0.88,
      envMapIntensity: 0.5,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const grid = new THREE.GridHelper(90, 60, 0x1b2a3a, 0x121c26);
  grid.position.y = 0.012;
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);
}

// --------------------------------------------------------------------------
// Canvas-texture label helper
// --------------------------------------------------------------------------

function makeLabel(worldWidth, worldHeight, pxWidth = 512, pxHeight = 256) {
  const cv = document.createElement('canvas');
  cv.width = pxWidth;
  cv.height = pxHeight;
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(worldWidth, worldHeight),
    new THREE.MeshBasicMaterial({
      map: tex, transparent: true, depthWrite: false, toneMapped: false,
    }),
  );
  return { mesh, ctx: cv.getContext('2d'), tex, w: pxWidth, h: pxHeight };
}

// --------------------------------------------------------------------------
// Vault construction
// --------------------------------------------------------------------------

const vaults = VAULTS.map((spec, i) => {
  const group = new THREE.Group();
  group.position.set(spec.pos[0], spec.pos[1], spec.pos[2]);
  scene.add(group);

  // plinth
  const plinth = new THREE.Mesh(
    new THREE.CylinderGeometry(spec.radius * 1.42, spec.radius * 1.55, 0.55, 48),
    new THREE.MeshStandardMaterial({
      color: 0x161d26, roughness: 0.3, metalness: 0.95, envMapIntensity: 0.8,
    }),
  );
  plinth.position.y = 0.275;
  plinth.castShadow = true;
  plinth.receiveShadow = true;
  group.add(plinth);

  // emissive ring set into the plinth, tinted to the currency
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(spec.radius * 1.25, 0.035, 8, 64),
    new THREE.MeshBasicMaterial({ color: spec.colour, toneMapped: false }),
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.57;
  group.add(ring);

  // glass shell
  const shell = new THREE.Mesh(
    new THREE.CylinderGeometry(spec.radius, spec.radius, spec.height, 64, 1, true),
    new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      transmission: 1.0,
      thickness: 0.9,
      ior: 1.46,
      roughness: 0.055,
      metalness: 0.0,
      clearcoat: 1.0,
      clearcoatRoughness: 0.07,
      envMapIntensity: 1.3,
      side: THREE.DoubleSide,
    }),
  );
  shell.position.y = 0.55 + spec.height / 2;
  group.add(shell);

  // metal collar at the top of the glass
  const collar = new THREE.Mesh(
    new THREE.CylinderGeometry(spec.radius * 1.07, spec.radius * 1.07, 0.3, 48),
    new THREE.MeshStandardMaterial({
      color: 0x1c242e, roughness: 0.28, metalness: 0.96, envMapIntensity: 0.9,
    }),
  );
  collar.position.y = 0.55 + spec.height + 0.1;
  collar.castShadow = true;
  group.add(collar);

  // liquid: a cylinder scaled on Y, so scale.y maps directly to pool fill
  const liquid = new THREE.Mesh(
    new THREE.CylinderGeometry(spec.radius * 0.94, spec.radius * 0.94, spec.height, 56),
    new THREE.MeshStandardMaterial({
      color: spec.colour,
      emissive: spec.colour,
      emissiveIntensity: 0.55,
      roughness: 0.22,
      metalness: 0.1,
      transparent: true,
      opacity: 0.92,
    }),
  );
  liquid.castShadow = true;
  group.add(liquid);

  // bright meniscus disc riding on top of the liquid
  const meniscus = new THREE.Mesh(
    new THREE.CircleGeometry(spec.radius * 0.94, 56),
    new THREE.MeshBasicMaterial({
      color: spec.colour,
      transparent: true,
      opacity: 0.85,
      toneMapped: false,
      side: THREE.DoubleSide,
    }),
  );
  meniscus.rotation.x = -Math.PI / 2;
  group.add(meniscus);

  // red shutter ring, shown while the corridor is paused
  const shutter = new THREE.Mesh(
    new THREE.TorusGeometry(spec.radius * 1.3, 0.07, 8, 64),
    new THREE.MeshBasicMaterial({
      color: 0xf43f5e, transparent: true, opacity: 0.0, toneMapped: false,
    }),
  );
  shutter.rotation.x = Math.PI / 2;
  shutter.position.y = 0.55 + spec.height * 0.5;
  shutter.visible = false;
  group.add(shutter);

  // label board in front of the plinth
  const label = makeLabel(spec.radius * 2.9, spec.radius * 1.45);
  label.mesh.position.set(0, 1.0, spec.radius * 1.62);
  group.add(label.mesh);

  return {
    spec, group, liquid, meniscus, shutter, label, ring,
    fill: 0.5, targetFill: 0.5, flash: 0, lastRate: null,
  };
});

// --------------------------------------------------------------------------
// Pipe manifold: each fiat vault to the USDT hub
// --------------------------------------------------------------------------

const hubSpec = VAULTS[HUB];
const pipes = [0, 1, 2].map((i) => {
  const spec = VAULTS[i];
  const start = new THREE.Vector3(
    spec.pos[0], 0.55 + spec.height + 0.1, spec.pos[2],
  );
  const end = new THREE.Vector3(
    hubSpec.pos[0], 0.55 + hubSpec.height + 0.1, hubSpec.pos[2],
  );
  const mid = start.clone().lerp(end, 0.5);
  mid.y += 2.6;

  const curve = new THREE.QuadraticBezierCurve3(start, mid, end);

  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 72, 0.085, 12, false),
    new THREE.MeshStandardMaterial({
      color: 0x27323e,
      emissive: spec.colour,
      emissiveIntensity: 0.0,
      roughness: 0.3,
      metalness: 0.92,
      envMapIntensity: 0.9,
    }),
  );
  tube.castShadow = true;
  scene.add(tube);

  // valve body where the pipe leaves the vault
  const valve = new THREE.Mesh(
    new THREE.SphereGeometry(0.17, 20, 16),
    new THREE.MeshStandardMaterial({
      color: 0x2a3542, roughness: 0.25, metalness: 0.95,
    }),
  );
  valve.position.copy(start);
  scene.add(valve);

  // travelling pulse, hidden until a rebalance fires
  const pulse = new THREE.Mesh(
    new THREE.SphereGeometry(0.2, 20, 16),
    new THREE.MeshBasicMaterial({ color: spec.colour, toneMapped: false }),
  );
  pulse.visible = false;
  scene.add(pulse);

  const glow = new THREE.PointLight(spec.colour, 0, 6);
  scene.add(glow);

  return { curve, tube, pulse, glow, t: 0, active: false, forward: true };
});

// --------------------------------------------------------------------------
// Spread gauge: bars above the hub, lit according to the quoted spread
// --------------------------------------------------------------------------

const spreadBars = [];
let spreadLabel;
{
  const n = 6;
  for (let i = 0; i < n; i += 1) {
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(0.17, 0.5, 0.17),
      new THREE.MeshBasicMaterial({ color: 0x1e2a36, toneMapped: false }),
    );
    const a = (-0.55 + (1.1 * i) / (n - 1)) * Math.PI;
    bar.position.set(
      hubSpec.pos[0] + Math.sin(a) * 2.5,
      0.55 + hubSpec.height + 1.25,
      hubSpec.pos[2] + Math.cos(a) * 2.5 * 0.35,
    );
    scene.add(bar);
    spreadBars.push(bar);
  }

  const gaugeLabel = makeLabel(3.0, 0.8, 512, 136);
  gaugeLabel.mesh.position.set(
    hubSpec.pos[0], 0.55 + hubSpec.height + 2.05, hubSpec.pos[2] + 0.9,
  );
  scene.add(gaugeLabel.mesh);
  spreadLabel = gaugeLabel;
}

// --------------------------------------------------------------------------
// Floor hologram: corridor arcs carrying client order flow
// --------------------------------------------------------------------------

const corridorArcs = [0, 1, 2].map((i) => {
  const spec = VAULTS[i];
  const start = new THREE.Vector3(spec.pos[0] * 0.78, 0.035, spec.pos[2] * 0.78);
  const end = new THREE.Vector3(hubSpec.pos[0], 0.035, hubSpec.pos[2] * 0.4);
  const mid = start.clone().lerp(end, 0.5);
  mid.x += (i - 1) * 0.6;

  const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
  const arc = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 48, 0.022, 8, false),
    new THREE.MeshBasicMaterial({
      color: spec.colour, transparent: true, opacity: 0.3, toneMapped: false,
    }),
  );
  scene.add(arc);

  const packet = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 14, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }),
  );
  packet.visible = false;
  scene.add(packet);

  return { curve, arc, packet, t: 0, active: false, forward: true };
});

{
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(2.0, 2.06, 96),
    new THREE.MeshBasicMaterial({
      color: 0x1d4e5e, transparent: true, opacity: 0.5,
      side: THREE.DoubleSide, toneMapped: false,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(hubSpec.pos[0], 0.03, hubSpec.pos[2] * 0.4);
  scene.add(ring);
}

// --------------------------------------------------------------------------
// FX wall: three sparklines of each currency against USD
// --------------------------------------------------------------------------

const fxWall = makeLabel(26, 9, 1536, 540);
fxWall.mesh.position.set(0, 6.6, -13.5);
fxWall.mesh.material.opacity = 0.95;
scene.add(fxWall.mesh);

const history = [[], [], []];

function drawFxWall(state) {
  const { ctx, w, h } = fxWall;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = 'rgba(8, 12, 17, 0.82)';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(120, 150, 180, 0.18)';
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, w - 2, h - 2);

  ctx.fillStyle = 'rgba(139, 152, 168, 0.95)';
  ctx.font = '600 22px ui-monospace, monospace';
  ctx.fillText('FX  vs  USD   —   % change since open', 28, 40);

  const padX = 28;
  const padTop = 62;
  const panelH = (h - padTop - 20) / 3;

  for (let i = 0; i < 3; i += 1) {
    const top = padTop + i * panelH;
    const series = history[i];

    ctx.strokeStyle = 'rgba(120, 150, 180, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padX, top + panelH * 0.5);
    ctx.lineTo(w - padX, top + panelH * 0.5);
    ctx.stroke();

    ctx.fillStyle = CSS_COLOUR[i];
    ctx.font = '700 26px ui-monospace, monospace';
    ctx.fillText(VAULTS[i].name, padX, top + panelH * 0.5 - 10);

    if (series.length > 1) {
      // symmetric scale so depreciation reads as a fall
      let span = 0.004;
      for (const v of series) span = Math.max(span, Math.abs(v));
      span *= 1.2;

      const x0 = padX + 92;
      const x1 = w - padX - 150;
      const yMid = top + panelH * 0.5;
      const yAmp = panelH * 0.36;

      ctx.beginPath();
      series.forEach((v, k) => {
        const x = x0 + ((x1 - x0) * k) / (SPARK_HISTORY - 1);
        const y = yMid - (v / span) * yAmp;
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = CSS_COLOUR[i];
      ctx.lineWidth = 2.6;
      ctx.stroke();

      const last = series[series.length - 1];
      const lastX = x0 + ((x1 - x0) * (series.length - 1)) / (SPARK_HISTORY - 1);
      const lastY = yMid - (last / span) * yAmp;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
      ctx.fillStyle = CSS_COLOUR[i];
      ctx.fill();

      ctx.fillStyle = last > 0.0005 ? '#f43f5e' : (last < -0.0005 ? '#22c55e' : '#8b98a8');
      ctx.font = '700 28px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${(last * 100).toFixed(2)}%`, w - padX - 18, yMid + 10);
      ctx.textAlign = 'left';

      if (state && state.vol_regime && state.vol_regime[i] > 3.0) {
        ctx.fillStyle = 'rgba(244, 63, 94, 0.9)';
        ctx.font = '700 17px ui-monospace, monospace';
        ctx.fillText('HIGH VOL', x0 - 2, top + 26);
      }
    }
  }

  fxWall.tex.needsUpdate = true;
}

// --------------------------------------------------------------------------
// Vault labels
// --------------------------------------------------------------------------

function fmtCompact(x) {
  const a = Math.abs(x);
  if (a >= 1e9) return `${(x / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(x / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${(x / 1e3).toFixed(1)}K`;
  return x.toFixed(0);
}

function drawVaultLabel(v, i, state) {
  const { ctx, w, h } = v.label;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = 'rgba(8, 12, 17, 0.7)';
  roundRect(ctx, 2, 2, w - 4, h - 4, 16);
  ctx.fill();
  ctx.strokeStyle = CSS_COLOUR[i];
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.fillStyle = CSS_COLOUR[i];
  ctx.font = '700 60px ui-monospace, monospace';
  ctx.fillText(v.spec.name, 26, 72);

  ctx.fillStyle = 'rgba(232, 238, 245, 0.95)';
  ctx.font = '600 40px ui-monospace, monospace';
  ctx.fillText(`$${fmtCompact(state.pools_usd[i])}`, 26, 128);

  ctx.fillStyle = 'rgba(139, 152, 168, 0.9)';
  ctx.font = '500 30px ui-monospace, monospace';
  if (i === HUB) {
    ctx.fillText('settlement rail', 26, 178);
  } else {
    const chg = state.rate_change[i] * 100;
    ctx.fillText(`${fmtCompact(state.balances_native[i])} @ ${state.rates[i].toFixed(1)}`, 26, 178);
    ctx.fillStyle = chg > 0 ? '#f43f5e' : '#22c55e';
    ctx.fillText(`${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%`, 26, 220);
  }

  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(139, 152, 168, 0.75)';
  ctx.font = '600 28px ui-monospace, monospace';
  ctx.fillText(`${(v.targetFill * 100).toFixed(0)}%`, w - 26, 72);
  ctx.textAlign = 'left';

  v.label.tex.needsUpdate = true;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawSpreadLabel(spread) {
  const { ctx, w, h } = spreadLabel;
  ctx.clearRect(0, 0, w, h);
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(139, 152, 168, 0.85)';
  ctx.font = '600 34px ui-monospace, monospace';
  ctx.fillText('QUOTED SPREAD', w / 2, 44);
  ctx.fillStyle = '#06b6d4';
  ctx.font = '700 60px ui-monospace, monospace';
  ctx.fillText(`${(spread * 100).toFixed(2)}%`, w / 2, 110);
  ctx.textAlign = 'left';
  spreadLabel.tex.needsUpdate = true;
}

// --------------------------------------------------------------------------
// The agent: an operator drone
//
// Embodying the policy makes it legible. Instead of numbers changing, you
// watch something fly to the column it has decided to act on and do the work.
// Every action maps to a destination and a gesture.
// --------------------------------------------------------------------------

const IDLE_POS = new THREE.Vector3(0, 9.4, 3.2);

const drone = new THREE.Group();
drone.position.copy(IDLE_POS);
scene.add(drone);

let droneParts;
{
  const shell = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.42, 1),
    new THREE.MeshStandardMaterial({
      color: 0x8fa4b8, roughness: 0.25, metalness: 0.95, envMapIntensity: 1.3,
    }),
  );
  shell.castShadow = true;
  drone.add(shell);

  // glowing core, tinted to whatever the drone is currently acting on
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.17, 20, 16),
    new THREE.MeshBasicMaterial({ color: 0x06b6d4, toneMapped: false }),
  );
  core.position.y = -0.3;
  drone.add(core);

  // counter-rotating gimbal rings
  const ringGeo = new THREE.TorusGeometry(0.62, 0.028, 8, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x06b6d4, transparent: true, opacity: 0.75, toneMapped: false,
  });
  const ringA = new THREE.Mesh(ringGeo, ringMat);
  ringA.rotation.x = Math.PI / 2;
  const ringB = new THREE.Mesh(ringGeo, ringMat.clone());
  ringB.rotation.y = Math.PI / 2;
  drone.add(ringA, ringB);

  // four thruster glows
  const thrusters = [];
  for (let i = 0; i < 4; i += 1) {
    const a = (i / 4) * Math.PI * 2;
    const t = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 10, 8),
      new THREE.MeshBasicMaterial({ color: 0x9beaff, toneMapped: false }),
    );
    t.position.set(Math.cos(a) * 0.52, -0.1, Math.sin(a) * 0.52);
    drone.add(t);
    thrusters.push(t);
  }

  const lamp = new THREE.PointLight(0x06b6d4, 6, 9);
  drone.add(lamp);

  // working beam, aimed at the target of the current action
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.055, 0.14, 1, 14, 1, true),
    new THREE.MeshBasicMaterial({
      color: 0x06b6d4, transparent: true, opacity: 0.0,
      toneMapped: false, side: THREE.DoubleSide, depthWrite: false,
    }),
  );
  scene.add(beam);

  droneParts = { shell, core, ringA, ringB, thrusters, lamp, beam };
}

const agent = {
  target: IDLE_POS.clone(),
  aimAt: null,        // world point the beam is pointing at, or null
  colour: 0x06b6d4,
  busy: 0,            // seconds of "working" animation remaining
  parts: droneParts,
};

const UP = new THREE.Vector3(0, 1, 0);
const _dir = new THREE.Vector3();

function aimBeam(beam, from, to) {
  _dir.subVectors(to, from);
  const len = _dir.length();
  if (len < 1e-4) return;
  beam.position.copy(from).addScaledVector(_dir, 0.5);
  beam.scale.set(1, len, 1);
  beam.quaternion.setFromUnitVectors(UP, _dir.clone().normalize());
}

function vaultTop(i) {
  const s = VAULTS[i];
  return new THREE.Vector3(s.pos[0], 0.55 + s.height * 0.72, s.pos[2]);
}

/** Point the drone at whatever this action operates on. */
function directAgent(state) {
  const a = state.action;
  const parts = agent.parts;

  // rebalance: hover over the source pool and draw from it
  if (state.rebalance) {
    const src = state.rebalance.src;
    const top = vaultTop(src);
    agent.target.set(top.x, top.y + 2.3, top.z + 1.5);
    agent.aimAt = top;
    agent.colour = VAULTS[src].colour;
    agent.busy = 0.9;
    return;
  }

  // spread: fly up to the gauge above the hub
  if (a === 7 || a === 8) {
    agent.target.set(0, 0.55 + hubSpec.height + 2.6, hubSpec.pos[2] + 2.2);
    agent.aimAt = new THREE.Vector3(
      hubSpec.pos[0], 0.55 + hubSpec.height + 1.25, hubSpec.pos[2],
    );
    agent.colour = 0x06b6d4;
    agent.busy = 0.7;
    return;
  }

  // pause: drop down to the corridor gate and seal it
  if (a >= 9 && a <= 11) {
    const i = a - 9;
    const s = VAULTS[i];
    agent.target.set(s.pos[0], 0.55 + s.height * 0.5 + 1.8, s.pos[2] + 2.0);
    agent.aimAt = new THREE.Vector3(s.pos[0], 0.55 + s.height * 0.5, s.pos[2]);
    agent.colour = 0xf43f5e;
    agent.busy = 1.1;
    return;
  }

  // resume: return to centre and pulse
  if (a === 12) {
    agent.target.copy(IDLE_POS);
    agent.aimAt = null;
    agent.colour = 0x22c55e;
    agent.busy = 0.6;
    return;
  }

  // hold: drift back to station
  agent.target.copy(IDLE_POS);
  agent.aimAt = null;
  agent.colour = 0x06b6d4;
  parts.beam.material.opacity = 0.0;
}

function animateAgent(dt, time) {
  const p = agent.parts;

  // ease toward the station the current action implies
  drone.position.lerp(agent.target, Math.min(1, dt * 2.6));
  drone.position.y += Math.sin(time * 2.1) * 0.012;

  p.ringA.rotation.z += dt * 1.5;
  p.ringB.rotation.x -= dt * 1.1;

  const c = new THREE.Color(agent.colour);
  p.core.material.color.lerp(c, Math.min(1, dt * 6));
  p.ringA.material.color.lerp(c, Math.min(1, dt * 6));
  p.ringB.material.color.lerp(c, Math.min(1, dt * 6));
  p.lamp.color.lerp(c, Math.min(1, dt * 6));

  const working = agent.busy > 0;
  if (working) agent.busy = Math.max(0, agent.busy - dt);

  // thrusters flare while repositioning
  const travel = drone.position.distanceTo(agent.target);
  const flare = 0.75 + Math.min(travel, 3) * 0.3 + (working ? 0.4 : 0);
  for (const t of p.thrusters) t.scale.setScalar(flare);
  p.lamp.intensity = 5 + (working ? 7 : 0) + Math.sin(time * 9) * 0.6;

  if (working && agent.aimAt) {
    p.beam.material.color.set(agent.colour);
    p.beam.material.opacity = 0.3 + 0.35 * Math.abs(Math.sin(time * 14));
    aimBeam(p.beam, drone.position, agent.aimAt);
  } else {
    p.beam.material.opacity = Math.max(0, p.beam.material.opacity - dt * 3);
  }
}

// --------------------------------------------------------------------------
// Post-processing
// --------------------------------------------------------------------------

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight), 0.62, 0.72, 0.82,
);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// --------------------------------------------------------------------------
// HUD
// --------------------------------------------------------------------------

const el = (id) => document.getElementById(id);
const feed = [];
let alertTimer = null;

function signed(x, digits = 2) {
  return `${x >= 0 ? '+' : ''}${x.toFixed(digits)}`;
}


// --------------------------------------------------------------------------
// Policy panel: the decision, not just the outcome.
// --------------------------------------------------------------------------
let brainRows = null;

function buildBrainRows(names) {
  const host = el('brainrows');
  if (!host) return null;
  host.innerHTML = '';
  const rows = names.map((raw) => {
    const name = raw.replace(/_/g, ' ');
    const row = document.createElement('div');
    row.className = 'brow';
    const label = document.createElement('div');
    label.textContent = name;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = '0%';
    bar.appendChild(fill);
    const value = document.createElement('div');
    value.className = 'v';
    value.textContent = '—';
    row.append(label, bar, value);
    host.appendChild(row);
    return { row, fill, value };
  });
  return rows;
}


// --------------------------------------------------------------------------
// Devaluation impact. The event the entire policy exists to anticipate should
// be felt, not merely logged: a short camera shake plus a red screen flash.
// --------------------------------------------------------------------------
let shake = 0;
let devaluationWasActive = false;

function triggerImpact() {
  shake = 1.0;
  const f = el('impact');
  if (!f) return;
  // Snap to full opacity, then let CSS fade it out.
  f.style.transition = 'none';
  f.style.opacity = '0.9';
  requestAnimationFrame(() => {
    f.style.transition = 'opacity 1100ms cubic-bezier(0.2, 0, 0.3, 1)';
    f.style.opacity = '0';
  });
}

function checkImpact(state) {
  const active = (state.devaluation || []).some(Boolean);
  if (active && !devaluationWasActive) triggerImpact();
  devaluationWasActive = active;
}

function applyShake(dt) {
  if (shake <= 0) return;
  // Applied after controls.update(), which rebuilds the camera position from
  // the orbit state each frame, so this offset never accumulates.
  const amp = shake * shake * 0.42;
  camera.position.x += (Math.random() - 0.5) * amp;
  camera.position.y += (Math.random() - 0.5) * amp;
  camera.position.z += (Math.random() - 0.5) * amp;
  camera.lookAt(controls.target);
  shake = Math.max(0, shake - dt * 1.9);
}


// --------------------------------------------------------------------------
// Playback controls and episode history.
//
// Commands travel back over the same socket that carries diagnostics. The
// server is authoritative: a click sends an intent and the button settles to
// whatever the next frame reports, so the UI cannot drift out of step with
// the simulation.
// --------------------------------------------------------------------------
let playbackPaused = false;
let playbackSpeed = 1.0;

function sendControl(cmd, value) {
  const payload = { type: 'control', cmd };
  if (value !== undefined) payload.value = value;
  try {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  } catch (err) {
    reportError('control', err);
  }
}

function renderPlayback() {
  const play = el('btnplay');
  const step = el('btnstep');
  const label = el('speedlabel');
  if (play) {
    play.textContent = playbackPaused ? 'Resume' : 'Pause';
    play.classList.toggle('paused', playbackPaused);
  }
  if (step) step.disabled = !playbackPaused;
  if (label) label.textContent = `${playbackSpeed.toFixed(1)}\u00d7`;
}

function bindControls() {
  const play = el('btnplay');
  const step = el('btnstep');
  const slower = el('btnslower');
  const faster = el('btnfaster');

  if (play) play.addEventListener('click', () => {
    playbackPaused = !playbackPaused;      // optimistic; the next frame confirms
    renderPlayback();
    sendControl('toggle');
  });
  if (step) step.addEventListener('click', () => sendControl('step'));
  if (slower) slower.addEventListener('click', () => {
    playbackSpeed = Math.max(0.25, playbackSpeed / 2);
    renderPlayback();
    sendControl('speed', playbackSpeed);
  });
  if (faster) faster.addEventListener('click', () => {
    playbackSpeed = Math.min(8, playbackSpeed * 2);
    renderPlayback();
    sendControl('speed', playbackSpeed);
  });

  // Space toggles, right-arrow steps -- the shortcuts anyone recording a
  // walkthrough will reach for.
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      if (play) play.click();
    } else if (e.code === 'ArrowRight' && playbackPaused) {
      e.preventDefault();
      sendControl('step');
    }
  });
  renderPlayback();
}

function syncPlayback(state) {
  const pb = state.playback;
  if (!pb) return;
  const changed = pb.paused !== playbackPaused || pb.speed !== playbackSpeed;
  playbackPaused = !!pb.paused;
  playbackSpeed = Number(pb.speed) || 1.0;
  if (changed) renderPlayback();
}

// --- completed weeks ------------------------------------------------------
const weeks = [];
let lastWeekRecorded = -1;

function updateHistory(state) {
  if (!state.done) return;
  const idx = state.episode_index;
  if (idx === undefined || idx === lastWeekRecorded) return;
  lastWeekRecorded = idx;

  weeks.unshift({
    n: idx,
    ret: state.episode_return,
    ok: state.verdict !== 'insolvent',
    equity: state.equity_usd / state.initial_equity_usd - 1,
  });
  if (weeks.length > 8) weeks.pop();

  const host = el('historyrows');
  if (!host) return;
  host.innerHTML = '';
  weeks.forEach((w) => {
    const row = document.createElement('div');
    row.className = 'hrow';
    const ret = typeof w.ret === 'number' ? w.ret.toFixed(1) : '—';
    row.innerHTML = `<span class="w">wk ${w.n}</span>`
      + `<span class="o ${w.ok ? 'ok' : 'bad'}">`
      + `${w.ok ? 'SURVIVED' : 'INSOLVENT'} ${(w.equity * 100).toFixed(2)}%</span>`
      + `<span class="s">${ret}</span>`;
    host.appendChild(row);
  });
}

function updateBrain(state) {
  const host = el('brainrows');
  const none = el('brainnone');
  const kindEl = el('brainkind');
  if (!host || !none) return;

  const names = state.action_names;
  const scores = state.policy_scores;
  if (!Array.isArray(scores) || !Array.isArray(names)
      || scores.length !== names.length) {
    host.style.display = 'none';
    none.style.display = 'block';
    if (kindEl) kindEl.textContent = 'RULE-BASED';
    return;
  }
  host.style.display = 'block';
  none.style.display = 'none';

  if (!brainRows) brainRows = buildBrainRows(names);
  if (!brainRows) return;

  const isQ = state.policy_score_kind === 'q';
  if (kindEl) kindEl.textContent = isQ ? 'Q-VALUES' : 'ACTION PROBABILITY';

  // For logits, a softmax is the policy's actual distribution. For Q-values a
  // softmax would be meaningless, so the bar is min-max scaled instead and the
  // label shows the raw action value.
  let display;
  let widths;
  if (isQ) {
    const lo = Math.min(...scores);
    const hi = Math.max(...scores);
    const span = hi - lo || 1;
    widths = scores.map((s) => (s - lo) / span);
    display = scores.map((s) => s.toFixed(1));
  } else {
    const hi = Math.max(...scores);
    const exps = scores.map((s) => Math.exp(s - hi));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    const probs = exps.map((e) => e / sum);
    widths = probs.slice();
    display = probs.map((p) => `${(p * 100).toFixed(0)}%`);
  }

  const chosen = scores.indexOf(Math.max(...scores));
  for (let i = 0; i < brainRows.length; i += 1) {
    const r = brainRows[i];
    r.fill.style.width = `${Math.max(0, Math.min(1, widths[i])) * 100}%`;
    r.value.textContent = display[i];
    r.row.classList.toggle('on', i === chosen);
  }
}

function updateHud(state) {
  const equity = state.equity_usd;
  const pnl = equity / state.initial_equity_usd - 1.0;

  el('equity').textContent = `$${equity.toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })}`;
  const pnlEl = el('pnl');
  pnlEl.textContent = `${signed(pnl * 100)}%`;
  pnlEl.className = `mono ${pnl >= 0 ? 'pos' : 'neg'}`;

  const ddEl = el('dd');
  ddEl.textContent = `${(state.drawdown * 100).toFixed(2)}%`;
  ddEl.className = `mono ${state.drawdown > 0.08 ? 'neg' : ''}`;

  el('spread').textContent = `${(state.spread * 100).toFixed(2)}%`;
  el('fees').textContent = `$${state.stats.cum_fees_usd.toFixed(0)}`;
  el('filled').textContent = `${state.stats.n_filled}`;

  const soEl = el('stockouts');
  soEl.textContent = `${state.stats.n_unfilled}`;
  soEl.className = `mono ${state.stats.n_unfilled > 0 ? 'neg' : 'pos'}`;

  el('stepcount').textContent = `step ${state.t} / ${state.episode_len}`;
  const rwEl = el('reward');
  rwEl.textContent = `reward ${signed(state.reward, 3)}`;
  rwEl.className = state.reward >= 0 ? 'pos' : 'neg';
  el('barfill').style.width = `${(state.t / state.episode_len) * 100}%`;

  if (state.policy) {
    el('polname').textContent = state.policy;
    el('agentname').textContent = `${state.policy} agent`;
  }

  // what the agent is doing, in words
  const idle = state.action_name === 'HOLD';
  el('doing').textContent = idle ? 'HOLDING POSITION' : state.action_name.replace(/_/g, ' ');
  el('doing').style.color = idle ? 'var(--dim)' : 'var(--usdt)';
  el('agentstate').textContent = idle ? 'MONITORING' : 'EXECUTING';

  // devaluation pressure gauges: the signal the policy is meant to trade on
  const pressure = state.pressure || [0, 0, 0];
  const hazard = state.hazard || [0, 0, 0];
  el('gauges').innerHTML = [0, 1, 2]
    .map((i) => {
      const p = Math.min(pressure[i] / 1.6, 1);
      const hot = pressure[i] > 0.85;
      const warm = pressure[i] > 0.5;
      const colour = hot ? '#f43f5e' : (warm ? '#f59e0b' : CSS_COLOUR[i]);
      // hazard is per-step; show it over the rest of the week instead
      const left = Math.max(state.episode_len - state.t, 1);
      const risk = (1 - Math.pow(1 - hazard[i], left)) * 100;
      return `<div class="gauge">
        <div class="top">
          <span style="color:${colour};font-weight:600">${VAULTS[i].name}</span>
          <span style="color:var(--dim)">${risk.toFixed(0)}% risk this week</span>
        </div>
        <div class="track"><div class="fill"
          style="width:${(p * 100).toFixed(0)}%;background:${colour}"></div></div>
      </div>`;
    })
    .join('');

  // devaluation klaxon
  const broke = (state.devaluation || []).findIndex(Boolean);
  if (broke >= 0) {
    el('alertccy').textContent = VAULTS[broke].name;
    el('alert').style.display = 'block';
    clearTimeout(alertTimer);
    alertTimer = setTimeout(() => { el('alert').style.display = 'none'; }, 3200);
  }

  // action feed
  if (state.action_name !== 'HOLD' || feed.length === 0) {
    feed.unshift({ t: state.t, a: state.action_name, r: state.reward });
    if (feed.length > FEED_LENGTH) feed.pop();
  } else {
    feed[0] = { t: state.t, a: state.action_name, r: state.reward };
  }
  el('feedlist').innerHTML = feed
    .map((f) => `<div><span class="t">${f.t}</span>`
      + `<span class="a">${f.a.replace(/_/g, ' ')}</span>`
      + `<span class="r ${f.r >= 0 ? 'pos' : 'neg'}">${signed(f.r, 2)}</span></div>`)
    .join('');

  // corridor legend
  el('corridors').innerHTML = [0, 1, 2]
    .map((i) => {
      const chg = state.rate_change[i] * 100;
      const cls = chg > 0 ? 'neg' : 'pos';
      const paused = state.paused[i]
        ? '<span class="pausedtag">PAUSED</span>' : '';
      return `<div class="corridor">
        <span class="dot" style="background:${CSS_COLOUR[i]}"></span>
        <span class="nm">${VAULTS[i].name}</span>
        <span class="rt">${state.rates[i].toFixed(1)}</span>
        <span class="dv ${cls}">${signed(chg)}%</span>
        ${paused}</div>`;
    })
    .join('');
}

function showVerdict(state) {
  const pnl = state.equity_usd / state.initial_equity_usd - 1.0;
  const insolvent = state.verdict === 'insolvent';
  el('vhead').textContent = insolvent ? 'Desk insolvent' : 'Trading week complete';
  el('vhead').style.color = insolvent ? '#f43f5e' : '#22c55e';
  el('vdetail').innerHTML = insolvent
    ? `Capital reserve breached at step <b>${state.t}</b>. `
      + `Equity <b>$${state.equity_usd.toFixed(0)}</b> `
      + `(<b>${signed(pnl * 100)}%</b>).`
    : `Equity <b>$${state.equity_usd.toFixed(0)}</b> `
      + `(<b>${signed(pnl * 100)}%</b>) · `
      + `<b>${state.stats.n_filled}</b> orders filled · `
      + `<b>${state.stats.n_unfilled}</b> stockouts · `
      + `max drawdown <b>${(state.drawdown * 100).toFixed(2)}%</b>.`;
  el('verdict').style.display = 'block';
}

function hideVerdict() {
  el('verdict').style.display = 'none';
}

// --------------------------------------------------------------------------
// State application
// --------------------------------------------------------------------------

let latest = null;

function applyState(state) {
  updateBrain(state);
  checkImpact(state);
  syncPlayback(state);
  updateHistory(state);
  // a new episode has started: clear the previous verdict and the FX history
  if (latest && state.t < latest.t) {
    hideVerdict();
    for (const series of history) series.length = 0;
    feed.length = 0;
  }
  latest = state;

  for (let i = 0; i < 4; i += 1) {
    const v = vaults[i];
    v.targetFill = THREE.MathUtils.clamp(state.pool_fill[i], 0.015, 1.0);

    // devaluation flash: a sharp adverse move lights the column red
    if (i < 3) {
      const r = state.rates[i];
      if (v.lastRate !== null && r / v.lastRate - 1.0 > 0.006) v.flash = 1.0;
      v.lastRate = r;

      const paused = state.paused[i];
      v.shutter.visible = paused;
      v.shutter.material.opacity = paused ? 0.75 : 0.0;
      corridorArcs[i].arc.material.opacity = paused ? 0.1 : 0.3;
      corridorArcs[i].arc.material.color.set(paused ? 0xf43f5e : VAULTS[i].colour);
    }

    drawVaultLabel(v, i, state);
  }

  // rebalance pulse along the manifold
  if (state.rebalance) {
    const { src, dst } = state.rebalance;
    const fiat = dst === HUB ? src : dst;
    if (fiat >= 0 && fiat < 3) {
      const p = pipes[fiat];
      p.active = true;
      p.forward = dst === HUB;
      p.t = 0;
      p.pulse.visible = true;
    }
  }

  // incoming client order travels the floor hologram
  if (state.pending_order) {
    const a = corridorArcs[state.pending_order.corridor];
    a.active = true;
    a.forward = state.pending_order.direction === 1;
    a.t = 0;
    a.packet.visible = true;
  }

  // spread gauge
  const lit = Math.round(((state.spread - 0.005) / (0.030 - 0.005)) * (spreadBars.length - 1));
  spreadBars.forEach((b, k) => {
    b.material.color.set(k <= lit ? 0x06b6d4 : 0x1e2a36);
    b.scale.y = k <= lit ? 1.0 + 0.5 * (k / spreadBars.length) : 0.65;
  });
  drawSpreadLabel(state.spread);

  // FX history
  for (let i = 0; i < 3; i += 1) {
    history[i].push(state.rate_change[i]);
    if (history[i].length > SPARK_HISTORY) history[i].shift();
  }
  drawFxWall(state);

  directAgent(state);
  updateHud(state);

  if (state.done) showVerdict(state);
}

// --------------------------------------------------------------------------
// Animation loop
// --------------------------------------------------------------------------

const clock = new THREE.Clock();

function animate() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const time = clock.elapsedTime;

  for (const v of vaults) {
    // ease the liquid toward its target so levels glide instead of jumping
    v.fill += (v.targetFill - v.fill) * Math.min(1, dt * 7.5);
    const height = v.spec.height;
    const level = Math.max(v.fill, 0.012) * height;

    v.liquid.scale.y = Math.max(v.fill, 0.012);
    v.liquid.position.y = 0.55 + level / 2;

    // gentle surface bob, so the liquid reads as liquid
    const bob = Math.sin(time * 1.5 + v.spec.pos[0]) * 0.016;
    v.meniscus.position.y = 0.55 + level + 0.012 + bob;

    // flash decay
    if (v.flash > 0) {
      v.flash = Math.max(0, v.flash - dt * 1.6);
      v.liquid.material.emissive.setHex(0xf43f5e);
      v.liquid.material.emissiveIntensity = 0.55 + v.flash * 2.4;
      v.group.position.x = v.spec.pos[0] + Math.sin(time * 48) * 0.035 * v.flash;
    } else {
      v.liquid.material.emissive.setHex(v.spec.colour);
      v.liquid.material.emissiveIntensity = 0.55;
      v.group.position.x = v.spec.pos[0];
    }

    // a draining pool pulses its plinth ring as a warning
    if (v.fill < 0.12) {
      v.ring.material.color.setHex(0xf43f5e);
      v.ring.material.opacity = 0.5 + 0.5 * Math.sin(time * 7);
      v.ring.material.transparent = true;
    } else {
      v.ring.material.color.setHex(v.spec.colour);
      v.ring.material.opacity = 1.0;
    }

    if (v.shutter.visible) {
      v.shutter.rotation.z = time * 0.8;
    }
  }

  // manifold transfer pulses
  for (const p of pipes) {
    if (p.active) {
      p.t += dt * 2.1;
      if (p.t >= 1) {
        p.t = 0;
        p.active = false;
        p.pulse.visible = false;
        p.glow.intensity = 0;
        p.tube.material.emissiveIntensity = 0;
      } else {
        const u = p.forward ? p.t : 1 - p.t;
        p.curve.getPointAt(u, p.pulse.position);
        p.glow.position.copy(p.pulse.position);
        p.glow.intensity = 9 * Math.sin(Math.PI * p.t);
        p.tube.material.emissiveIntensity = 1.5 * Math.sin(Math.PI * p.t);
        const s = 0.8 + 0.5 * Math.sin(Math.PI * p.t);
        p.pulse.scale.setScalar(s);
      }
    }
  }

  // client order packets on the floor
  for (const a of corridorArcs) {
    if (a.active) {
      a.t += dt * 1.5;
      if (a.t >= 1) {
        a.t = 0;
        a.active = false;
        a.packet.visible = false;
      } else {
        const u = a.forward ? a.t : 1 - a.t;
        a.curve.getPointAt(u, a.packet.position);
        a.packet.scale.setScalar(0.7 + 0.5 * Math.sin(Math.PI * a.t));
      }
    }
  }

  animateAgent(dt, time);

  controls.update();
  applyShake(dt);
  composer.render();
}

// A throw inside requestAnimationFrame silently stops the loop and freezes the
// canvas, so the failure is caught, reported once, and the loop keeps running.
let renderFailed = false;

function frame() {
  try {
    animate();
  } catch (err) {
    if (!renderFailed) {
      renderFailed = true;
      reportError('render loop', err);
    }
  }
  requestAnimationFrame(frame);
}

bindControls();

frame();

// --------------------------------------------------------------------------
// Resize
// --------------------------------------------------------------------------

window.addEventListener('resize', () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloom.setSize(w, h);
});

// --------------------------------------------------------------------------
// Websocket transport
// --------------------------------------------------------------------------

const wsPort = new URLSearchParams(window.location.search).get('ws') || '8766';
const wsUrl = `ws://${window.location.hostname}:${wsPort}`;
let attempts = 0;
// The control bar posts commands back on whichever socket is live.
let socket = null;

function connect() {
  el('waithint').textContent = attempts
    ? `retrying ${wsUrl} (attempt ${attempts + 1})`
    : wsUrl;

  const ws = new WebSocket(wsUrl);
  socket = ws;

  ws.onopen = () => {
    attempts = 0;
    el('waiting').style.display = 'none';
    el('verdict').style.display = 'none';
    // relay anything that already failed, plus everything from here on
    errorSink = (payload) => ws.send(payload);
    for (const entry of errorLog) ws.send(JSON.stringify(entry));
  };

  ws.onmessage = (ev) => {
    try {
      applyState(JSON.parse(ev.data));
    } catch (err) {
      reportError('applyState', err);
    }
  };

  ws.onclose = () => {
    attempts += 1;
    errorSink = null;
    if (socket === ws) socket = null;
    if (!latest) el('waiting').style.display = 'flex';
    setTimeout(connect, Math.min(500 * attempts, 4000));
  };

  ws.onerror = () => ws.close();
}

connect();

/* Panther simulation view: URDF forward kinematics drawn on a <canvas>, no libraries.
 *
 * Joint positions come from GET /api/v1/robotics/joints (streamed by the ROS 2 bridge from
 * /joint_states, the lift and the gripper states). Point the bridge at the HEADBands mock sim
 * (ROS_DOMAIN_ID=77) to watch the sim, or at the robot (15) to watch the robot. "Demo" animates
 * the joints locally when no bridge is streaming. This view only reads; it never commands.
 *
 * Links are drawn with the panther_sim stylized meshes (assets/meshes, WebGL on #simMeshCanvas
 * over the floor grid). Without WebGL, or until they load: rods between joint frames, orange
 * spheres on moving joints, a cyan dot on the RGB-D camera.
 */
(() => {
  const canvas = document.getElementById("simCanvas");
  if (!canvas) return;
  const meshCanvas = document.getElementById("simMeshCanvas");
  const statusEl = document.getElementById("simStatus");
  const demoToggle = document.getElementById("simDemoToggle");
  const resetBtn = document.getElementById("simResetView");

  const LIFT_TOP = 0.8; // panther_examples.robot.MAX_LIFT: 0 = bottom, 0.8 m = top
  const DEFAULT_VIEW = { yaw: 35, pitch: 18, dist: 2.6, target: [0.2, 0, 0.9] };
  const view = { ...DEFAULT_VIEW };
  const COLORS = {
    left: "#3c96eb", right: "#5ac85a", head: "#c878dc", body: "#c3c3c8",
    joint: "#f58c28", camera: "#28dce6", grid: "#7aa866", gridAxis: "#f6f2ea", base: "#7c8088",
  };

  let model = null;
  let positions = {};
  let source = "waiting";
  let lastAge = null;
  let demoStart = performance.now();

  // ------------------------------------------------------------------ math
  const mat = {
    identity: () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    mul(a, b) {
      const o = new Array(16).fill(0);
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
          for (let k = 0; k < 4; k++) o[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
      return o;
    },
    fromRT(R, t) {
      return [R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1];
    },
    rpy(r, p, y) {
      const [cr, sr, cp, sp, cy, sy] = [Math.cos(r), Math.sin(r), Math.cos(p), Math.sin(p), Math.cos(y), Math.sin(y)];
      return [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
              sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
              -sp, cp * sr, cp * cr];
    },
    axisAngle([x, y, z], a) {
      const c = Math.cos(a), s = Math.sin(a), t = 1 - c;
      return [t * x * x + c, t * x * y - s * z, t * x * z + s * y,
              t * x * y + s * z, t * y * y + c, t * y * z - s * x,
              t * x * z - s * y, t * y * z + s * x, t * z * z + c];
    },
    apply(m, [x, y, z]) {
      return [m[0] * x + m[1] * y + m[2] * z + m[3], m[4] * x + m[5] * y + m[6] * z + m[7], m[8] * x + m[9] * y + m[10] * z + m[11]];
    },
  };
  const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const nums = (s, d) => (s || d).trim().split(/\s+/).map(Number);

  // ------------------------------------------------------------------ URDF
  function parseUrdf(text) {
    const doc = new DOMParser().parseFromString(text, "application/xml");
    const joints = [];
    for (const j of doc.querySelectorAll("robot > joint")) {
      const origin = j.querySelector("origin");
      let xyz = nums(origin && origin.getAttribute("xyz"), "0 0 0");
      const rpy = nums(origin && origin.getAttribute("rpy"), "0 0 0");
      let axis = nums(j.querySelector("axis") && j.querySelector("axis").getAttribute("xyz"), "1 0 0");
      const n = Math.hypot(...axis) || 1;
      axis = axis.map((v) => v / n);
      const lim = j.querySelector("limit");
      let lower = lim ? Number(lim.getAttribute("lower") || 0) : 0;
      let upper = lim ? Number(lim.getAttribute("upper") || 0) : 0;
      let type = j.getAttribute("type") || "fixed";
      const name = j.getAttribute("name");
      // Same lift change as panther_sim's mock_sim.launch.py: vendor axis 0 0 -1 (0-0.6 m)
      // becomes +z, 0 (bottom) to 0.8 m (top), with the origin lowered by the full travel.
      if (name === "body_joint" && axis[2] < -0.99) {
        type = "prismatic"; axis = [0, 0, 1]; lower = 0; upper = LIFT_TOP;
        xyz = [xyz[0], xyz[1], xyz[2] - LIFT_TOP];
      }
      const mimicEl = j.querySelector("mimic");
      joints.push({
        name, type, lower, upper, axis,
        parent: j.querySelector("parent").getAttribute("link"),
        child: j.querySelector("child").getAttribute("link"),
        origin: mat.fromRT(mat.rpy(...rpy), xyz),
        mimic: mimicEl ? [mimicEl.getAttribute("joint"), Number(mimicEl.getAttribute("multiplier") || 1),
                          Number(mimicEl.getAttribute("offset") || 0)] : null,
      });
    }
    const children = new Set(joints.map((j) => j.child));
    const root = (joints.find((j) => !children.has(j.parent)) || {}).parent;
    const byParent = {};
    joints.forEach((j) => (byParent[j.parent] = byParent[j.parent] || []).push(j));
    const ordered = [];
    const stack = [root];
    while (stack.length) {
      for (const j of byParent[stack.pop()] || []) { ordered.push(j); stack.push(j.child); }
    }
    // Vendor mesh paths (file:///.../share/mh3_description/meshes/...) -> /assets/meshes/...;
    // the Odin1 lidar (another package) has no stand-in of its own: base_link's covers it.
    const meshes = {};
    for (const l of doc.querySelectorAll("robot > link")) {
      const m = l.querySelector("visual > geometry > mesh");
      const f = m && m.getAttribute("filename");
      if (f && f.includes("/share/mh3_description/")) meshes[l.getAttribute("name")] = "/assets/" + f.split("/share/mh3_description/")[1];
    }
    return { root, joints: ordered, meshes };
  }

  function forwardKinematics(m, q) {
    const frames = { [m.root]: mat.identity() };
    for (const j of m.joints) {
      const parent = frames[j.parent];
      if (!parent) continue;
      const v = j.mimic ? (q[j.mimic[0]] || 0) * j.mimic[1] + j.mimic[2] : q[j.name] || 0;
      let motion = mat.identity();
      if (j.type === "revolute" || j.type === "continuous") motion = mat.fromRT(mat.axisAngle(j.axis, v), [0, 0, 0]);
      else if (j.type === "prismatic") motion = mat.fromRT(I3, j.axis.map((a) => a * v));
      frames[j.child] = mat.mul(mat.mul(parent, j.origin), motion);
    }
    return frames;
  }

  function demoPositions(m, tSec) {
    const out = {};
    m.joints.forEach((j, i) => {
      if (j.mimic || !(j.type === "revolute" || j.type === "prismatic")) return;
      let { lower: lo, upper: hi } = j;
      if (hi <= lo) { lo = -0.5; hi = 0.5; }
      const amp = Math.min(((hi - lo) / 2) * 0.6, 0.7);
      out[j.name] = (lo + hi) / 2 + amp * Math.sin(tSec * 0.5 + i * 0.7);
    });
    return out;
  }

  // ------------------------------------------------------------------ drawing
  const group = (link) => {
    const n = link.toLowerCase();
    if (n.includes("_la") || n.includes("_lf") || n.startsWith("left")) return "left";
    if (n.includes("_ra") || n.includes("_rf") || n.startsWith("right")) return "right";
    if (n.includes("head") || n.includes("rgbd") || n.includes("eye")) return "head";
    return "body";
  };

  function cameraBasis(h) {
    const yaw = (view.yaw * Math.PI) / 180, pitch = (view.pitch * Math.PI) / 180;
    const t = view.target;
    const eye = [t[0] + view.dist * Math.cos(pitch) * Math.cos(yaw), t[1] + view.dist * Math.cos(pitch) * Math.sin(yaw),
                 t[2] + view.dist * Math.sin(pitch)];
    const norm = (v) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const fwd = norm(t.map((v, i) => v - eye[i]));
    const right = norm(cross(fwd, [0, 0, 1]));
    const up = cross(right, fwd);
    const f = h / 2 / Math.tan((45 * Math.PI) / 360);
    return { eye, fwd, right, up, f };
  }

  function makeCamera(w, h) {
    const { eye, fwd, right, up, f } = cameraBasis(h);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return (p) => {
      const d = p.map((v, i) => v - eye[i]);
      const z = dot(d, fwd);
      if (z < 0.05) return null;
      return [w / 2 + (f * dot(d, right)) / z, h / 2 - (f * dot(d, up)) / z, z, f];
    };
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 480;
    const cssH = canvas.clientHeight || Math.round(cssW * 0.75);
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const w = canvas.width, h = canvas.height;
    const ctx = canvas.getContext("2d");
    drawScenery(ctx, w, h);
    if (!model) return;

    const project = makeCamera(w, h);
    const prims = [];
    const line = (a, b, color, thick) => {
      const pa = project(a), pb = project(b);
      if (!pa || !pb) return;
      const z = (pa[2] + pb[2]) / 2;
      prims.push({ z, kind: "line", pa, pb, color, width: Math.max(1, (pa[3] * thick) / z) });
    };
    const ball = (p, color, r) => {
      const pp = project(p);
      if (pp) prims.push({ z: pp[2], kind: "ball", pp, color, r: Math.max(2, (pp[3] * r) / pp[2]) });
    };

    for (let k = -2; k <= 2.001; k += 0.5) {
      const c = Math.abs(k) < 1e-6 ? COLORS.gridAxis : COLORS.grid;
      line([k, -2, 0], [k, 2, 0], c, 0.004);
      line([-2, k, 0], [2, k, 0], c, 0.004);
    }
    let frames = forwardKinematics(model, positions);
    if (basePose) { // a previewed base move carries the whole robot
      const T = mat.fromRT(mat.axisAngle([0, 0, 1], basePose[2]), [basePose[0], basePose[1], 0]);
      frames = Object.fromEntries(Object.entries(frames).map(([k, F]) => [k, mat.mul(T, F)]));
    }
    if (!drawMeshes(frames)) drawSticks(frames, line, ball); // the meshes replace the stick figure
    paint(ctx, prims);
  }

  // 3D scenery after the Leukerbad Robotics Hackathon poster: a coral sky, faceted Alpine peaks (snow-white
  // and blue faces), a hazier range behind, and a navy ground disk that ends at the peaks' feet.
  // Everything is in world coordinates, so it turns and zooms with the robot.
  const GRASS = "#2f4f2a"; // dark alpine meadow
  const GROUND_R = 25; // radius of the ground disk; every peak's foot tucks just under its edge
  const rand = (seed) => { let x = seed; return () => (x = (x * 16807) % 2147483647) / 2147483647; };
  const at = (ang, r, z) => [r * Math.cos(ang), r * Math.sin(ang), z];

  function face(out, tri, lit, shade) {
    const [a, b, c] = tri;
    const u = b.map((v, k) => v - a[k]), v = c.map((x, k) => x - a[k]);
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const cx = a[0] + b[0] + c[0], cy = a[1] + b[1] + c[1];
    if (n[0] * cx + n[1] * cy > 0) n = n.map((x) => -x); // the side facing the robot, at the ring's centre
    // Poster lighting: light from the same side of every peak, as seen from the robot, so each one shows a
    // snow side and a blue side from any view. tang = how much the facet faces that side; nz = how flat it is.
    const len = Math.hypot(...n) || 1, cl = Math.hypot(cx, cy) || 1;
    const tang = (n[0] * -cy + n[1] * cx) / cl / len, nz = n[2] / len;
    out.push({ pts: tri, color: tang > 0.08 || nz > 0.9 ? lit : shade });
  }

  // One peak: a jagged ridge row, two jittered mid-slope rows and a foot row on the ground nearer the robot,
  // stitched into triangles so every facet catches the light differently. Concave sides make it pointy.
  function peak(out, rnd, ang, halfW, R, H, depth, lit, shade) {
    const N = 11, ROWS = [[1, 0, 0.8, 0.45], [0.55, 0.35, 0.3, 0.3], [0.2, 0.7, 0.1, 0.2], [0, 1, 0, 0]];
    const rows = ROWS.map(() => []);
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1), edge = i === 0 || i === N - 1;
      const shape = Math.pow(1 - Math.abs(2 * t - 1), 1.5);
      const a = ang + (t - 0.5) * 2 * halfW;
      ROWS.forEach(([hk, dk, hMin, hVar], r) => {
        const wob = r && !edge ? (rnd() - 0.5) * halfW * 0.2 : 0;
        const z = r === ROWS.length - 1 || edge ? -0.3 : -0.3 + H * shape * hk * (hMin / hk + rnd() * hVar / Math.max(hk, 0.2));
        rows[r].push(at(a + wob, R - depth * dk, r === 0 && !edge ? -0.3 + H * shape * (0.75 + rnd() * 0.45) : z));
      });
    }
    for (let r = 0; r < rows.length - 1; r++) {
      for (let i = 0; i < N - 1; i++) {
        const a = rows[r][i], b = rows[r][i + 1], c = rows[r + 1][i], d = rows[r + 1][i + 1];
        face(out, [a, b, d], lit, shade);
        face(out, [a, d, c], lit, shade);
      }
    }
  }

  function range(R, count, hMin, hMax, depth, lit, shade, seed) {
    const rnd = rand(seed), out = [];
    for (let i = 0; i < count; i++) {
      const ang = ((i + rnd() * 0.6) / count) * Math.PI * 2;
      const halfW = (Math.PI / count) * (0.85 + rnd() * 0.5); // neighbours just overlap
      const r = R * (0.92 + rnd() * 0.16), height = hMin + rnd() * (hMax - hMin);
      // the foot row reaches just under the ground disk's edge, so no peak floats or shows a gap below it
      peak(out, rnd, ang, halfW, r, height, Math.max(depth, r - GROUND_R + 0.5), lit, shade);
    }
    return out;
  }

  // Props on the meadow: chalets and pines, placed once at random (fixed seed) in a ring around the robot,
  // kept apart from each other and clear of the robot's area (the camera orbits up to 7 m out).
  // Row-major 4x4 transforms, as the FK frames. sx = sy = k, sz = k * stretch.
  const placed = []; // [x, y, keep-out radius] of everything placed so far
  function scatter(n, rMin, rMax, kMin, kMax, stretch, gap, seed, turn) {
    const rnd = rand(seed), out = [];
    for (let tries = 0; out.length < n && tries < n * 40; tries++) {
      const a = rnd() * Math.PI * 2, r = rMin + rnd() * (rMax - rMin), k = kMin + rnd() * (kMax - kMin);
      const x = r * Math.cos(a), y = r * Math.sin(a);
      if (placed.some(([px, py, pr]) => Math.hypot(px - x, py - y) < pr + gap)) continue;
      placed.push([x, y, gap]);
      const yaw = turn ? a + Math.PI + (rnd() - 0.5) * 0.8 : rnd() * Math.PI * 2; // chalets face the robot
      const c = Math.cos(yaw) * k, s = Math.sin(yaw) * k, kz = k * (1 + (rnd() - 0.5) * stretch);
      out.push([c, -s, 0, x, s, c, 0, y, 0, 0, kz, -0.05, 0, 0, 0, 1]);
    }
    return out;
  }
  const CHALETS = scatter(30, 10, 20, 1.6, 2.2, 0, 1.3, 11, true); // model 1 m tall
  const TREES = scatter(70, 8.5, 23, 0.9, 1.7, 0.5, 0.45, 23, false); // pine 1.7 m tall
  const COWS = scatter(16, 8.5, 18, 0.45, 0.55, 0, 0.6, 31, false); // cow 1 m tall, 1.76 m long

  const SCENERY = [
    ...range(55, 13, 4.5, 7.5, 12, "#f6e4dc", "#8cc6e4", 7), // back range, hazier
    ...range(38, 17, 2.8, 5, 9, "#f6f2ea", "#67b6de", 3), // main range: snow and blue shadow
  ];

  function drawScenery(ctx, w, h) {
    const { fwd, up, f } = cameraBasis(h);
    // horizon: where a far point straight ahead, at camera height, lands on screen
    const hl = Math.hypot(fwd[0], fwd[1]) || 1e-6;
    const hf = [fwd[0] / hl, fwd[1] / hl, 0];
    const depth = hf[0] * fwd[0] + hf[1] * fwd[1];
    const yH = Math.min(Math.max(h / 2 - (f * (hf[0] * up[0] + hf[1] * up[1])) / Math.max(depth, 1e-3), 0), h);
    const g = ctx.createLinearGradient(0, 0, 0, Math.max(yH, 1));
    g.addColorStop(0, "#ec6a65");
    g.addColorStop(1, "#f29490");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h); // sky everywhere; the ground disk is drawn over it after the peaks

    const project = makeCamera(w, h);
    const tris = [];
    for (const t of SCENERY) {
      const pp = t.pts.map(project);
      if (pp.every(Boolean)) tris.push({ pp, color: t.color, z: Math.max(pp[0][2], pp[1][2], pp[2][2]) });
    }
    tris.sort((a, b) => b.z - a.z); // far to near
    ctx.lineWidth = 1;
    for (const t of tris) {
      ctx.fillStyle = ctx.strokeStyle = t.color; // the stroke hides hairline seams between facets
      ctx.beginPath();
      ctx.moveTo(t.pp[0][0], t.pp[0][1]);
      ctx.lineTo(t.pp[1][0], t.pp[1][1]);
      ctx.lineTo(t.pp[2][0], t.pp[2][1]);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    // Ground: a dark-green disk out to just before the peaks' feet. Its far edge is a curve across the view;
    // everything below it is ground, so between and past the peaks there is only sky.
    const edge = [];
    for (let k = 0; k < 360; k++) {
      const p = project(at((k / 360) * Math.PI * 2, GROUND_R, 0));
      if (p && p[2] > 0.5) edge.push(p);
    }
    ctx.fillStyle = GRASS;
    if (!edge.length) { ctx.fillRect(0, 0, w, h); return; } // looking straight down: all ground
    edge.sort((a, b) => a[0] - b[0]);
    ctx.beginPath();
    ctx.moveTo(Math.min(0, edge[0][0]), h);
    ctx.lineTo(Math.min(0, edge[0][0]), edge[0][1]);
    for (const p of edge) ctx.lineTo(p[0], p[1]);
    ctx.lineTo(Math.max(w, edge[edge.length - 1][0]), edge[edge.length - 1][1]);
    ctx.lineTo(Math.max(w, edge[edge.length - 1][0]), h);
    ctx.closePath();
    ctx.fill();
  }

  function drawSticks(frames, line, ball) {
    const base = frames.base_link || frames[model.root];
    const corners = [];
    for (const z of [0.06, 0.28]) for (const y of [-0.21, 0.21]) for (const x of [-0.25, 0.25]) corners.push(mat.apply(base, [x, y, z]));
    [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) =>
      line(corners[a], corners[b], COLORS.base, 0.012));

    for (const j of model.joints) {
      const fp = frames[j.parent], fc = frames[j.child];
      if (!fp || !fc) continue;
      if (j.parent === "base_link" && j.child !== "body_link") continue; // inside the base box
      const a = [fp[3], fp[7], fp[11]], b = [fc[3], fc[7], fc[11]];
      if (Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) > 0.02) {
        line(a, b, COLORS[group(j.child)], j.child === "body_link" ? 0.07 : 0.025);
      }
      if ((j.type === "revolute" || j.type === "prismatic") && !j.mimic) ball(b, COLORS.joint, 0.035);
      if (j.child.toLowerCase().includes("rgbd")) ball(b, COLORS.camera, 0.03);
    }
  }

  function paint(ctx, prims) {
    prims.sort((p, q) => q.z - p.z);
    ctx.lineCap = "round";
    for (const p of prims) {
      if (p.kind === "line") {
        ctx.strokeStyle = p.color;
        ctx.lineWidth = p.width;
        ctx.beginPath();
        ctx.moveTo(p.pa[0], p.pa[1]);
        ctx.lineTo(p.pb[0], p.pb[1]);
        ctx.stroke();
      } else {
        ctx.fillStyle = p.color;
        ctx.strokeStyle = "#141414";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p.pp[0], p.pp[1], p.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  // ------------------------------------------------------------------ meshes (WebGL)
  const gl = meshCanvas && meshCanvas.getContext("webgl", { alpha: true, antialias: true });
  const glMeshes = {}; // link -> { buf, count }
  const chaletParts = []; // { buf, count, color }: one per colour of assets/meshes/chalet (drawn at each of CHALETS)
  const treeParts = []; // the same for the low-poly pine built by pineMesh()
  const cowParts = []; // and for assets/meshes/cow (drawn at each of COWS)
  const PROPS = [{ parts: chaletParts, at: CHALETS }, { parts: treeParts, at: TREES }, { parts: cowParts, at: COWS }];

  // A low-poly pine, 1.7 m tall: a hexagonal trunk and three stacked six-sided cones.
  // Interleaved [x y z nx ny nz] per vertex, flat normals per facet, as parseStl gives.
  function pineMesh() {
    const tri = (out, a, b, c) => {
      const u = b.map((v, k) => v - a[k]), v = c.map((x, k) => x - a[k]);
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const l = Math.hypot(...n) || 1;
      for (const p of [a, b, c]) out.push(...p, n[0] / l, n[1] / l, n[2] / l);
    };
    const ring = (r, z, k, sides) => [r * Math.cos((k / sides) * Math.PI * 2), r * Math.sin((k / sides) * Math.PI * 2), z];
    const trunk = [], leaves = [];
    for (let k = 0; k < 6; k++) {
      const a0 = ring(0.08, 0, k, 6), a1 = ring(0.08, 0, k + 1, 6), b0 = ring(0.08, 0.45, k, 6), b1 = ring(0.08, 0.45, k + 1, 6);
      tri(trunk, a0, a1, b1);
      tri(trunk, a0, b1, b0);
    }
    for (const [r, z0, z1] of [[0.55, 0.3, 1.0], [0.44, 0.65, 1.35], [0.3, 1.0, 1.7]]) {
      for (let k = 0; k < 6; k++) {
        tri(leaves, ring(r, z0, k, 6), ring(r, z0, k + 1, 6), [0, 0, z1]);
        tri(leaves, ring(r, z0, k, 6), ring(r, z0, k + 1, 6), [0, 0, z0]); // underside, seen from low views
      }
    }
    return [[new Float32Array(trunk), "#6b4a2b"], [new Float32Array(leaves), "#2f6b45"]];
  }
  let glProg = null;

  function initGl() {
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };
    glProg = gl.createProgram();
    gl.attachShader(glProg, sh(gl.VERTEX_SHADER, `
      attribute vec3 pos; attribute vec3 nrm;
      uniform mat4 model; uniform vec3 eye, fwd, right, up; uniform vec2 scale;
      varying float shade;
      void main() {
        vec3 d = (model * vec4(pos, 1.0)).xyz - eye;
        float z = dot(d, fwd);
        vec3 n = normalize(mat3(model) * nrm);
        shade = 0.35 + 0.65 * abs(dot(n, normalize(up - fwd)));
        // near 0.05 m, far 50 m
        gl_Position = vec4(dot(d, right) * scale.x, dot(d, up) * scale.y, z * (50.05 / 49.95) - 5.0 / 49.95, z);
      }`));
    gl.attachShader(glProg, sh(gl.FRAGMENT_SHADER, `
      precision mediump float; uniform vec3 color; varying float shade;
      void main() { gl_FragColor = vec4(color * shade, 1.0); }`));
    gl.linkProgram(glProg);
    if (!gl.getProgramParameter(glProg, gl.LINK_STATUS)) { glProg = null; return; }
    gl.enable(gl.DEPTH_TEST);
  }

  // Binary STL -> interleaved [x y z nx ny nz] per vertex (facet normal on all three).
  function parseStl(buf) {
    const dv = new DataView(buf);
    const n = dv.getUint32(80, true);
    if (buf.byteLength < 84 + n * 50) throw new Error("short STL");
    const out = new Float32Array(n * 18);
    for (let i = 0; i < n; i++) {
      const o = 84 + i * 50;
      const nrm = [dv.getFloat32(o, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)];
      for (let v = 0; v < 3; v++) {
        const k = (i * 3 + v) * 6;
        for (let c = 0; c < 3; c++) {
          out[k + c] = dv.getFloat32(o + 12 + v * 12 + c * 4, true);
          out[k + 3 + c] = nrm[c];
        }
      }
    }
    return out;
  }

  function loadMeshes(m) {
    if (!gl) return;
    initGl();
    if (!glProg) return;
    const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16) / 255);
    for (const [link, url] of Object.entries(m.meshes)) {
      fetch(url)
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText))))
        .then((buf) => {
          const data = parseStl(buf);
          const b = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, b);
          gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
          glMeshes[link] = { buf: b, count: data.length / 6, color: hex(COLORS[group(link)]) };
        })
        .catch(() => {}); // a missing part leaves a gap; the rest still draws
    }
    for (const [data, color] of pineMesh()) {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      treeParts.push({ buf: b, count: data.length / 6, color: hex(color) });
    }
    // Models converted to one STL per colour plus a <name>.json listing them (see the json for source and licence):
    // Swiss chalets, "Chalet" by Poly by Google (poly.pizza/m/6ufSK2nQBc6, CC BY 3.0), and Brown Swiss cows,
    // "Cow" by Quaternius (poly.pizza/m/26zM1outCr, CC0). A missing file only leaves that prop out.
    const loadProp = (name, parts) =>
      fetch(`/assets/meshes/${name}/${name}.json`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
        .then((c) => {
          for (const part of c.parts) {
            fetch(`/assets/meshes/${name}/${part.file}`)
              .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText))))
              .then((buf) => {
                const data = parseStl(buf);
                const b = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, b);
                gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
                parts.push({ buf: b, count: data.length / 6, color: hex(part.color) });
              })
              .catch(() => {});
          }
        })
        .catch(() => {});
    loadProp("chalet", chaletParts);
    loadProp("cow", cowParts);
  }

  // Draws every loaded link mesh at its FK frame. False (draw the stick figure) until the first one loads.
  function drawMeshes(frames) {
    if (!glProg || (!Object.keys(glMeshes).length && !PROPS.some((p) => p.parts.length))) {
      if (meshCanvas) meshCanvas.hidden = true;
      return false;
    }
    meshCanvas.hidden = false;
    const w = canvas.width, h = canvas.height;
    if (meshCanvas.width !== w || meshCanvas.height !== h) { meshCanvas.width = w; meshCanvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(glProg);
    const { eye, fwd, right, up, f } = cameraBasis(h);
    const u = (n) => gl.getUniformLocation(glProg, n);
    gl.uniform3fv(u("eye"), eye);
    gl.uniform3fv(u("fwd"), fwd);
    gl.uniform3fv(u("right"), right);
    gl.uniform3fv(u("up"), up);
    gl.uniform2f(u("scale"), (2 * f) / w, (2 * f) / h);
    const pos = gl.getAttribLocation(glProg, "pos"), nrm = gl.getAttribLocation(glProg, "nrm");
    gl.enableVertexAttribArray(pos);
    gl.enableVertexAttribArray(nrm);
    for (const [link, mesh] of Object.entries(glMeshes)) {
      const F = frames[link];
      if (!F) continue;
      // row-major 4x4 -> the column-major order WebGL wants
      gl.uniformMatrix4fv(u("model"), false, [0, 1, 2, 3].flatMap((c) => [F[c], F[4 + c], F[8 + c], F[12 + c]]));
      gl.uniform3fv(u("color"), mesh.color);
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buf);
      gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 24, 0);
      gl.vertexAttribPointer(nrm, 3, gl.FLOAT, false, 24, 12);
      gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
    }
    for (const { parts, at: places } of PROPS) {
      for (const M of places) {
        gl.uniformMatrix4fv(u("model"), false, [0, 1, 2, 3].flatMap((c) => [M[c], M[4 + c], M[8 + c], M[12 + c]]));
        for (const part of parts) {
          gl.uniform3fv(u("color"), part.color);
          gl.bindBuffer(gl.ARRAY_BUFFER, part.buf);
          gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 24, 0);
          gl.vertexAttribPointer(nrm, 3, gl.FLOAT, false, 24, 12);
          gl.drawArrays(gl.TRIANGLES, 0, part.count);
        }
      }
    }
    return Object.keys(glMeshes).length > 0; // props alone: the robot still needs its stick figure
  }

  let lastStatusAt = 0;
  function setStatus() {
    if (!statusEl) return;
    const now = performance.now();
    if (now - lastStatusAt < 500) return; // text twice a second, not every animation frame
    lastStatusAt = now;
    const q = positions;
    const fmt = (v) => (typeof v === "number" ? `${((v * 180) / Math.PI).toFixed(0)}°` : "unknown");
    const lift = typeof q.body_joint === "number" ? `${q.body_joint.toFixed(2)} m` : "unknown";
    const label = `3D view of the robot's pose. Lift ${lift}, head ${fmt(q.joint_head0)} and ${fmt(q.joint_head1)}.`;
    if (canvas.getAttribute("aria-label") !== label) canvas.setAttribute("aria-label", label);
    const reported = model
      ? model.joints.filter((j) => !j.mimic && (j.type === "revolute" || j.type === "prismatic") && j.name in positions).length
      : 0;
    const total = model ? model.joints.filter((j) => !j.mimic && (j.type === "revolute" || j.type === "prismatic")).length : 0;
    if (source === "preview" && preview) statusEl.textContent =
      `Preview: ${preview.name} (${preview.total.toFixed(1)} s, looping). Not sent to the robot.`;
    else if (source === "demo") statusEl.textContent = `Demo animation · ${reported}/${total} joints`;
    else if (source === "bridge") statusEl.textContent =
      `Live from ROS 2 bridge · ${reported}/${total} joints · ${lastAge !== null ? `${lastAge.toFixed(1)} s old` : ""}`;
    else if (source === "stale") statusEl.textContent =
      `No fresh joint states (last ${lastAge !== null ? `${lastAge.toFixed(0)} s` : "never"}) · showing last pose`;
    else statusEl.textContent = "Waiting for joint states from the ROS 2 bridge. Tick Demo to preview.";
  }

  // ------------------------------------------------------------------ data
  async function pollJoints() {
    if (document.hidden || preview || (demoToggle && demoToggle.checked)) return;
    try {
      const res = await fetch("/api/v1/robotics/joints");
      if (!res.ok) return;
      const data = await res.json();
      if (data.available) {
        positions = data.positions || {};
        lastAge = data.age_seconds;
        source = lastAge !== null && lastAge > 3 ? "stale" : "bridge";
      } else {
        source = "waiting";
      }
    } catch (_) {
      source = "waiting";
    }
  }

  // ------------------------------------------------------------------ motion preview
  // Plays pending motions (MotionPreview: joint points and/or a base path) one after the other, each from
  // where the last one ended, then loops after a short pause. Each step first eases from that pose to its
  // first point at a safe speed, as the bridge retimes it (retime_from_current_pose). Display only.
  const LEAD_SPEED = { body_joint: 0.05, joint_lf: 0.5, joint_rf: 0.5 }; // m/s or strokes/s; others 0.45 rad/s
  let preview = null; // { steps, total, start, name }
  let basePose = null; // [x, y, yaw] of the robot while previewing a base move, else null

  const composeBase = (b, d) => {
    const c = Math.cos(b[2]), s = Math.sin(b[2]);
    return [b[0] + c * d[0] - s * d[1], b[1] + s * d[0] + c * d[1], b[2] + d[2]];
  };

  function interpRows(rows, t) { // rows [[t, ...values]] with increasing t -> values at t (held past the ends)
    if (t <= rows[0][0]) return rows[0].slice(1);
    for (let k = 1; k < rows.length; k++) {
      if (t <= rows[k][0]) {
        const a = rows[k - 1], b = rows[k], s = b[0] > a[0] ? (t - a[0]) / (b[0] - a[0]) : 1;
        return a.slice(1).map((v, i) => v + s * (b[i + 1] - v));
      }
    }
    return rows[rows.length - 1].slice(1);
  }

  function startPreview(list) {
    const steps = [];
    const pose = { ...positions };
    let base = [0, 0, 0], total = 0;
    for (const p of [].concat(list || [])) {
      if (!p) continue;
      const names = p.joint_names || [], pts = p.points || [], path = p.base_path || [];
      if (!pts.length && !path.length) continue;
      let rows = null;
      if (pts.length && names.length) {
        const start = names.map((n, i) => (n in pose ? pose[n] : pts[0].positions[i]));
        const need = Math.max(0, ...names.map((n, i) => Math.abs(pts[0].positions[i] - start[i]) / (LEAD_SPEED[n] || 0.45)));
        const lead = Math.max(0, need - pts[0].t);
        rows = [[0, ...start], ...pts.map((pt) => [pt.t + lead, ...pt.positions])];
      }
      const baseRows = path.length ? [[0, 0, 0, 0], ...path] : null;
      const dur = Math.max(rows ? rows[rows.length - 1][0] : 0, baseRows ? baseRows[baseRows.length - 1][0] : 0);
      steps.push({ names, rows, baseRows, base0: base.slice(), from: { ...pose }, t0: total, dur });
      if (rows) names.forEach((n, i) => (pose[n] = rows[rows.length - 1][i + 1]));
      if (baseRows) base = composeBase(base, baseRows[baseRows.length - 1].slice(1));
      total += dur;
    }
    if (!steps.length) return false;
    const label = [].concat(list).filter(Boolean).map((p) => p.name).join(" → ");
    preview = { steps, total, start: performance.now(), name: label };
    if (demoToggle) demoToggle.checked = false;
    lastStatusAt = 0;
    return true;
  }

  function previewState(nowMs) {
    const t = ((nowMs - preview.start) / 1000) % (preview.total + 1.5); // loop with a short pause at the end
    let step = preview.steps[preview.steps.length - 1];
    for (const s of preview.steps) if (t < s.t0 + s.dur) { step = s; break; }
    const local = Math.min(t - step.t0, step.dur);
    const q = { ...step.from };
    if (step.rows) interpRows(step.rows, local).forEach((v, i) => (q[step.names[i]] = v));
    const base = step.baseRows ? composeBase(step.base0, interpRows(step.baseRows, local)) : step.base0;
    return { q, base: base[0] || base[1] || base[2] ? base : null };
  }

  // Seconds until the preview has played through once (0 when done or not previewing).
  const previewRemaining = () => (preview ? Math.max(0, preview.total - (performance.now() - preview.start) / 1000) : 0);

  function stopPreview() {
    preview = null;
    basePose = null;
    source = "waiting";
    lastStatusAt = 0;
    pollJoints();
  }

  function frame() {
    if (preview && model) {
      const st = previewState(performance.now());
      positions = st.q;
      basePose = st.base;
      source = "preview";
    } else if (demoToggle && demoToggle.checked && model) {
      positions = demoPositions(model, (performance.now() - demoStart) / 1000);
      source = "demo";
    }
    draw();
    setStatus();
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------ interaction
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    view.yaw = drag.yaw - (e.clientX - drag.x) * 0.4;
    view.pitch = Math.max(0, Math.min(80, drag.pitch + (e.clientY - drag.y) * 0.3));
  });
  canvas.addEventListener("pointerup", () => (drag = null));
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    view.dist = Math.max(1.2, Math.min(7, view.dist * (e.deltaY > 0 ? 1.1 : 0.9)));
  }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 15 : 5;
    const actions = {
      ArrowLeft: () => (view.yaw += step),
      ArrowRight: () => (view.yaw -= step),
      ArrowUp: () => (view.pitch = Math.min(80, view.pitch + step)),
      ArrowDown: () => (view.pitch = Math.max(0, view.pitch - step)),
      "+": () => (view.dist = Math.max(1.2, view.dist * 0.9)),
      "=": () => (view.dist = Math.max(1.2, view.dist * 0.9)),
      "-": () => (view.dist = Math.min(7, view.dist * 1.1)),
      0: () => Object.assign(view, DEFAULT_VIEW),
    };
    const act = actions[e.key];
    if (act) {
      e.preventDefault();
      act();
    }
  });
  if (resetBtn) resetBtn.addEventListener("click", () => Object.assign(view, DEFAULT_VIEW));
  if (demoToggle) demoToggle.addEventListener("change", () => {
    if (demoToggle.checked) preview = null;
    demoStart = performance.now();
    if (!demoToggle.checked) { positions = {}; source = "waiting"; pollJoints(); }
  });

  fetch("/assets/robot_description.urdf")
    .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
    .then((text) => { model = parseUrdf(text); loadMeshes(model); })
    .catch((err) => { if (statusEl) statusEl.textContent = `Could not load the robot model: ${err.message}`; });

  setInterval(pollJoints, 200);
  requestAnimationFrame(frame);
  window.PantherRobotView = {
    parseUrdf, forwardKinematics,
    preview: startPreview, // (MotionPreview or a list of them, played in order) -> false if nothing moves
    stopPreview,
    isPreviewing: () => Boolean(preview),
    previewRemaining,
  };
})();

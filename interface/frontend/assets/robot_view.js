/* Panther simulation view: URDF forward kinematics drawn on a <canvas>, no libraries.
 *
 * Joint positions come from GET /api/v1/robotics/joints (streamed by the ROS 2 bridge from
 * /joint_states, the lift and the gripper states). Point the bridge at the HEADBands mock sim
 * (ROS_DOMAIN_ID=77) to watch the sim, or at the robot (15) to watch the robot. "Demo" animates
 * the joints locally when no bridge is streaming. This view only reads; it never commands.
 *
 * Shapes are the same stand-ins RViz shows without the vendor meshes: rods between joint
 * frames, orange spheres on moving joints, a cyan dot on the RGB-D camera.
 */
(() => {
  const canvas = document.getElementById("simCanvas");
  if (!canvas) return;
  const statusEl = document.getElementById("simStatus");
  const demoToggle = document.getElementById("simDemoToggle");
  const resetBtn = document.getElementById("simResetView");

  const LIFT_TOP = 0.8; // panther_examples.robot.MAX_LIFT: 0 = bottom, 0.8 m = top
  const DEFAULT_VIEW = { yaw: 35, pitch: 18, dist: 2.6, target: [0.2, 0, 0.9] };
  const view = { ...DEFAULT_VIEW };
  const COLORS = {
    left: "#3c96eb", right: "#5ac85a", head: "#c878dc", body: "#c3c3c8",
    joint: "#f58c28", camera: "#28dce6", grid: "#334155", gridAxis: "#475569", base: "#7c8088",
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
    return { root, joints: ordered };
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

  function makeCamera(w, h) {
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
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "#1e293b");
    grad.addColorStop(1, "#0b1220");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
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
    const frames = forwardKinematics(model, positions);
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

  function setStatus() {
    if (!statusEl) return;
    const reported = model
      ? model.joints.filter((j) => !j.mimic && (j.type === "revolute" || j.type === "prismatic") && j.name in positions).length
      : 0;
    const total = model ? model.joints.filter((j) => !j.mimic && (j.type === "revolute" || j.type === "prismatic")).length : 0;
    if (source === "demo") statusEl.textContent = `Demo animation · ${reported}/${total} joints`;
    else if (source === "bridge") statusEl.textContent =
      `Live from ROS 2 bridge · ${reported}/${total} joints · ${lastAge !== null ? `${lastAge.toFixed(1)} s old` : ""}`;
    else if (source === "stale") statusEl.textContent =
      `No fresh joint states (last ${lastAge !== null ? `${lastAge.toFixed(0)} s` : "never"}) · showing last pose`;
    else statusEl.textContent = "Waiting for joint states from the ROS 2 bridge. Tick Demo to preview.";
  }

  // ------------------------------------------------------------------ data
  async function pollJoints() {
    if (document.hidden || (demoToggle && demoToggle.checked)) return;
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

  function frame() {
    if (demoToggle && demoToggle.checked && model) {
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
  if (resetBtn) resetBtn.addEventListener("click", () => Object.assign(view, DEFAULT_VIEW));
  if (demoToggle) demoToggle.addEventListener("change", () => {
    demoStart = performance.now();
    if (!demoToggle.checked) { positions = {}; source = "waiting"; pollJoints(); }
  });

  fetch("/assets/panther.urdf")
    .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
    .then((text) => { model = parseUrdf(text); })
    .catch((err) => { if (statusEl) statusEl.textContent = `Could not load the robot model: ${err.message}`; });

  setInterval(pollJoints, 200);
  requestAnimationFrame(frame);
  window.PantherRobotView = { parseUrdf, forwardKinematics }; // for debugging in the console
})();

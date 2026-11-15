/* Manual panel: one slider per joint (sent on release; the robot times the move at a safe speed) and base buttons
 * that jog while held. Checked by the backend and again by the bridge. The page resends a held base button every
 * 0.1 s and the bridge stops the base 0.2 s after the last one, so a closed tab or lost link stops it. */
(() => {
  const root = document.getElementById("manualControls");
  const baseEl = document.getElementById("manualBase");
  const statusEl = document.getElementById("manualStatus");
  const drawer = document.getElementById("manualDrawer");
  const toggle = document.getElementById("manualToggle");
  const closeBtn = document.getElementById("manualClose");
  if (!root) return;

  const arm = (side) => Array.from({ length: 8 }, (_, i) => `joint_${side}a${i}`);
  const GROUPS = [
    ["Head", ["joint_head0", "joint_head1"]],
    ["Lift", ["body_joint"]],
    ["Grippers", ["joint_lf", "joint_rf"]],
    ["Left arm (rad)", arm("l")],
    ["Right arm (rad)", arm("r")],
  ];
  const LABELS = {
    joint_head0: "Pitch (rad, + down)",
    joint_head1: "Yaw (rad, + left)",
    body_joint: "Height (m, 0 bottom)",
    joint_lf: "Left (0 open, 1 closed)",
    joint_rf: "Right (0 open, 1 closed)",
  };
  // shortcut: base speeds are master_gui's (BASE_LIN, BASE_ANG), not vendor limits; raise once tried on the robot.
  const BASE_LIN = 0.2;
  const BASE_ANG = 0.4;
  const BASE_BUTTONS = [
    ["⟲", "Turn left", [0, 0, 1]],
    ["↑", "Forward", [1, 0, 0]],
    ["⟳", "Turn right", [0, 0, -1]],
    ["←", "Left", [0, 1, 0]],
    ["↓", "Back", [-1, 0, 0]],
    ["→", "Right", [0, -1, 0]],
  ];
  const sliders = {}; // joint -> { input, now, set, mark, lo, hi, dragging, pending: { target, until } | null }
  const fmt = (v) => (v === undefined ? "–" : v.toFixed(2));

  function say(text) {
    if (statusEl.textContent !== text) statusEl.textContent = text;
  }

  async function send(body) {
    try {
      const res = await fetch("/api/v1/robotics/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) return true;
      const data = await res.json().catch(() => ({}));
      say(`Not sent: ${typeof data.detail === "string" ? data.detail : res.statusText}`);
    } catch (_) {
      say("Not sent: no connection to the server.");
    }
    return false;
  }

  function buildSliders(limits) {
    for (const [title, names] of GROUPS) {
      const group = document.createElement("fieldset");
      group.className = "manual-group";
      const legend = document.createElement("legend");
      legend.textContent = title;
      group.appendChild(legend);
      for (const name of names) {
        const [lo, hi] = limits[name];
        const input = document.createElement("input");
        input.type = "range";
        input.id = `manual-${name}`;
        input.min = lo;
        input.max = hi;
        input.step = name === "body_joint" ? 0.005 : 0.01;
        input.disabled = true; // until the robot reports this joint
        const now = document.createElement("output");
        now.value = "–";
        const set = document.createElement("output");
        set.htmlFor = input.id;
        set.value = "–";
        const vals = document.createElement("span");
        vals.className = "manual-vals";
        vals.append("now ", now, " · set ", set);
        const row = document.createElement("span");
        row.className = "manual-row";
        row.append(LABELS[name] || name, vals);
        const mark = document.createElement("span");
        mark.className = "manual-now-mark";
        mark.hidden = true;
        const track = document.createElement("span");
        track.className = "manual-track";
        track.append(input, mark);
        const label = document.createElement("label");
        label.htmlFor = input.id;
        label.append(row, track);
        group.appendChild(label);

        const s = { input, now, set, mark, lo, hi, dragging: false, pending: null };
        sliders[name] = s;
        input.addEventListener("input", () => {
          s.dragging = true;
          set.value = fmt(Number(input.value));
        });
        // "change" fires on release (mouse, touch) and on each arrow key press, as master_gui sends on release
        input.addEventListener("change", async () => {
          s.dragging = false;
          const target = Number(input.value);
          s.pending = { target, until: Date.now() + 20000 }; // the slider waits for the robot to get there
          if (await send({ joints: { [name]: target } })) say(`${LABELS[name] || name} → ${target.toFixed(2)}`);
        });
      }
      root.appendChild(group);
    }
  }

  // Sliders follow the robot's measured pose, except one being dragged or still moving to where it was let go.
  async function sync() {
    if (document.hidden || !drawer.classList.contains("open")) return;
    try {
      const res = await fetch("/api/v1/robotics/joints");
      if (!res.ok) return;
      const data = await res.json();
      const fresh = data.available && data.age_seconds !== null && data.age_seconds < 3;
      const pos = fresh ? data.positions : {};
      for (const [name, s] of Object.entries(sliders)) {
        const q = pos[name];
        s.input.disabled = q === undefined;
        s.now.value = fmt(q);
        s.mark.hidden = q === undefined;
        if (q === undefined) continue;
        const f = Math.min(Math.max((q - s.lo) / (s.hi - s.lo), 0), 1);
        s.mark.style.left = `calc(8px + ${f} * (100% - 16px))`;
        if (s.dragging) continue;
        if (s.pending && Math.abs(q - s.pending.target) > 0.02 && Date.now() < s.pending.until) continue;
        s.pending = null;
        s.input.value = q;
        s.set.value = fmt(q);
      }
      if (!fresh) say("Waiting for the robot's joint positions…");
      else if (statusEl.textContent.startsWith("Waiting")) say("");
    } catch (_) {}
  }

  // ---- base: resend every 0.1 s while held; zero on release, blur or hidden tab
  let jog = null;
  let timer = null;

  function jogStart(dir) {
    jog = [BASE_LIN * dir[0], BASE_LIN * dir[1], BASE_ANG * dir[2]];
    clearInterval(timer);
    send({ base: jog });
    timer = setInterval(() => send({ base: jog }), 100);
  }

  function jogStop() {
    if (!jog) return;
    jog = null;
    clearInterval(timer);
    timer = null;
    send({ base: [0, 0, 0] });
  }

  for (const [glyph, name, dir] of BASE_BUTTONS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn btn-secondary manual-jog";
    b.textContent = glyph;
    b.setAttribute("aria-label", `${name} (hold)`);
    b.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      b.setPointerCapture(e.pointerId);
      jogStart(dir);
    });
    for (const ev of ["pointerup", "pointercancel", "lostpointercapture", "blur"]) b.addEventListener(ev, jogStop);
    b.addEventListener("keydown", (e) => {
      if ((e.key === " " || e.key === "Enter") && !e.repeat) {
        e.preventDefault();
        jogStart(dir);
      }
    });
    b.addEventListener("keyup", (e) => (e.key === " " || e.key === "Enter") && jogStop());
    b.addEventListener("contextmenu", (e) => e.preventDefault()); // long press on touch screens
    baseEl.appendChild(b);
  }
  window.addEventListener("blur", jogStop);
  document.addEventListener("visibilitychange", jogStop);

  // ---- drawer: slides in from the right; closed, it is inert (not focusable) and nothing is polled
  function setOpen(open) {
    if (!open) jogStop();
    // start below the top bar so the emergency stop stays visible and clickable
    const bar = document.querySelector(".topbar");
    drawer.style.top = `${Math.max(0, bar ? bar.getBoundingClientRect().bottom : 0)}px`;
    drawer.classList.toggle("open", open);
    drawer.inert = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) {
      sync();
      closeBtn.focus();
    } else if (drawer.contains(document.activeElement)) {
      toggle.focus();
    }
  }
  toggle.addEventListener("click", () => setOpen(!drawer.classList.contains("open")));
  closeBtn.addEventListener("click", () => setOpen(false));
  drawer.addEventListener("keydown", (e) => e.key === "Escape" && setOpen(false));

  (async () => {
    try {
      const res = await fetch("/api/v1/robotics/manual");
      if (!res.ok) return say(`Manual control unavailable (${res.status}).`);
      buildSliders((await res.json()).limits);
      sync();
      setInterval(sync, 250);
    } catch (_) {
      say("Manual control unavailable: no connection to the server.");
    }
  })();
})();

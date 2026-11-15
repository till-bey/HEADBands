/* Live Panther camera feed for the Web UI.
 *
 * Repeatedly calls POST /api/v1/robotics/camera/capture (the same on-demand path as
 * "Trigger Robot Camera"): the backend asks the ROS 2 bridge, which answers from its buffered
 * /camera/color/image_raw frame. One request at a time, so a slow link never queues frames.
 * Starting the feed may auto-launch the Orbbec driver on the robot (bridge camera_manager);
 * the bridge shuts it down again after its idle timeout once the feed stops.
 */
(() => {
  const img = document.getElementById("cameraFeed");
  const toggle = document.getElementById("cameraLiveToggle");
  const statusEl = document.getElementById("cameraStatus");
  const placeholder = document.getElementById("cameraPlaceholder");
  if (!img || !toggle) return;

  const MIN_INTERVAL_MS = 350; // about 3 frames/s at most
  const RETRY_MS = 2000;
  let running = false;
  let frames = [];

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function setRunning(on) {
    running = on;
    toggle.textContent = on ? "■ Stop live camera" : "▶ Start live camera";
    toggle.classList.toggle("btn-secondary", on);
    toggle.classList.toggle("btn-primary", !on);
    if (on) loop();
    else setStatus("Stopped.");
  }

  async function loop() {
    while (running) {
      if (document.hidden) {
        setStatus("Paused while this tab is hidden.");
        await new Promise((r) => setTimeout(r, 1000));
        continue;
      }
      const started = performance.now();
      try {
        const res = await fetch("/api/v1/robotics/camera/capture", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: "Web UI live camera view", max_age_seconds: 0.3 }),
        });
        const data = await res.json().catch(() => ({}));
        if (!running) break;
        if (!res.ok) {
          setStatus(`Camera unavailable: ${String(data.detail || res.statusText).replace(/\.+$/, "")}. Retrying...`);
          await new Promise((r) => setTimeout(r, RETRY_MS));
          continue;
        }
        const f = data.frame;
        img.src = `data:${f.mime_type || "image/jpeg"};base64,${f.jpeg_base64}`;
        img.classList.remove("hidden");
        if (placeholder) placeholder.classList.add("hidden");
        const now = performance.now();
        frames = frames.filter((t) => now - t < 5000);
        frames.push(now);
        const fps = frames.length > 1 ? ((frames.length - 1) * 1000) / (now - frames[0]) : 0;
        const depth = f.center_depth_m !== null && f.center_depth_m !== undefined
          ? ` · centre depth ${Number(f.center_depth_m).toFixed(2)} m` : "";
        setStatus(`${f.source} · ${f.width}×${f.height} · ${fps.toFixed(1)} fps${depth}`);
      } catch (err) {
        setStatus(`Camera request failed: ${err.message}. Retrying...`);
        await new Promise((r) => setTimeout(r, RETRY_MS));
        continue;
      }
      const wait = MIN_INTERVAL_MS - (performance.now() - started);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
  }

  toggle.addEventListener("click", () => setRunning(!running));
  setRunning(false);
  setStatus("Press Start to stream /camera/color/image_raw from the robot.");
})();

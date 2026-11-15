/* Two views of one page: Simple (default, "/" or "#simple": mic, confirm, camera, simulation) and
 * Full ("#full": every panel and the manual control drawer). Other hashes (#main skip link) keep the view.
 * Loaded last, so the camera and manual scripts have set up their buttons. */
(() => {
  const link = document.getElementById("modeToggle");

  function apply() {
    let full;
    if (location.hash === "#full") full = true;
    else if (location.hash === "" || location.hash === "#simple") full = false;
    else return;
    document.body.classList.toggle("mode-full", full);
    link.textContent = full ? "Simple view" : "Full view";
    link.href = full ? "#simple" : "#full";
    // Simple view has no camera button: the feed starts by itself (the Full view keeps Start/Stop)
    const cam = document.getElementById("cameraLiveToggle");
    if (!full && cam && cam.getAttribute("aria-pressed") !== "true") cam.click();
    const drawer = document.getElementById("manualDrawer");
    if (!full && drawer && drawer.classList.contains("open")) document.getElementById("manualClose").click();
  }

  addEventListener("hashchange", apply);
  apply();
})();

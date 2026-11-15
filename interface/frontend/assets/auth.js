/* Page password (the backend's API_AUTH_TOKEN). Sent as a bearer token on every /api/ call and as ?token= on the
 * voice WebSocket; asked for when the server answers 401, remembered in this browser. Loaded before the other scripts. */
(() => {
  const KEY = "robotApiPassword";
  const dialog = document.getElementById("authDialog");
  const input = document.getElementById("authPassword");
  const button = document.getElementById("authBtn");
  let password = "";
  try {
    password = localStorage.getItem(KEY) || "";
  } catch (_) {}
  let asking = null;
  let declined = false; // after Cancel, only the Password button asks again (no pop-up on every poll)

  function ask() {
    if (!asking) {
      asking = new Promise((resolve) => {
        input.value = "";
        dialog.addEventListener(
          "close",
          () => {
            asking = null;
            const ok = dialog.returnValue === "ok" && input.value !== "";
            declined = !ok;
            if (ok) {
              password = input.value;
              try {
                localStorage.setItem(KEY, password);
              } catch (_) {}
            }
            resolve(ok);
          },
          { once: true },
        );
        dialog.returnValue = "";
        dialog.showModal();
      });
    }
    return asking;
  }

  const plainFetch = window.fetch.bind(window);
  window.fetch = async (url, opts = {}) => {
    if (typeof url !== "string" || !url.startsWith("/api/")) return plainFetch(url, opts);
    for (;;) {
      const headers = new Headers(opts.headers || {});
      if (password) headers.set("Authorization", `Bearer ${password}`);
      const res = await plainFetch(url, { ...opts, headers });
      if (res.status !== 401 || declined || !(await ask())) return res;
    }
  };

  window.apiPassword = () => password;
  button.addEventListener("click", () => {
    declined = false;
    ask().then((ok) => ok && window.location.reload()); // reconnects the voice socket with the new password
  });
})();

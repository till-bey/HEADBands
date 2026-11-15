/* Four-sided proximity display. Missing/stale data is never shown as clear. */
(() => {
  const sides = ['front', 'right', 'rear', 'left'];
  const closeDistance = 0.30;
  const warningDistance = 0.60;

  function classify(reading) {
    if (!reading.fresh || !Number.isFinite(reading.distance_m)) return 'unknown';
    if (reading.status === 'too_close') return 'danger';
    if (reading.status === 'beyond_range') return reading.distance_m >= warningDistance ? 'clear' : 'unknown';
    if (reading.status !== 'valid') return 'unknown';
    if (reading.distance_m <= closeDistance) return 'danger';
    return reading.distance_m <= warningDistance ? 'warning' : 'clear';
  }

  function groupReadings(readings, mapping = {}) {
    return Object.fromEntries(sides.map(side => {
      const explicit = Object.hasOwn(mapping, side);
      const names = explicit ? mapping[side] : readings.filter(r => r.side === side).map(r => r.topic);
      const selected = names.map(topic => readings.find(r => r.topic === topic));
      const levels = selected.map(r => r ? classify(r) : 'unknown');
      const level = levels.includes('danger') ? 'danger' : levels.includes('warning') ? 'warning'
        : !levels.length || levels.includes('unknown') ? 'unknown' : 'clear';
      const measured = selected.filter(r => r && r.fresh && Number.isFinite(r.distance_m)
        && ['valid', 'too_close', 'beyond_range'].includes(r.status));
      measured.sort((a, b) => a.distance_m - b.distance_m);
      const nearest = measured[0];
      const prefix = nearest?.status === 'too_close' ? '≤ ' : nearest?.status === 'beyond_range' ? '> ' : '';
      return [side, {level, label: nearest ? `${prefix}${nearest.distance_m.toFixed(2)} m` : '—',
        topics: names, partial: levels.includes('unknown')}];
    }));
  }

  if (typeof module !== 'undefined') module.exports = {classify, groupReadings};
  if (typeof document === 'undefined') return;
  const storageKey = 'navigationUltrasoundSides';
  let mapping = {};
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
    for (const side of sides) if (Array.isArray(saved?.[side])) mapping[side] = saved[side].filter(t => typeof t === 'string');
  } catch (_) {}
  let lastState = {};
  let avoidanceBusy = false;

  async function configureAvoidance(enabled, requestedMapping) {
    if (avoidanceBusy || typeof globalThis.configureUltrasoundAvoidance !== 'function') return;
    avoidanceBusy = true;
    render(lastState);
    try { await globalThis.configureUltrasoundAvoidance(enabled, requestedMapping); }
    finally { avoidanceBusy = false; render(lastState); }
  }

  function render(state) {
    lastState = state;
    // While avoidance is enabled, every tab displays the server's actual map.
    if (state.avoidance_enabled && state.avoidance_mapping) mapping = structuredClone(state.avoidance_mapping);
    const toggle = document.getElementById('avoidance-toggle');
    toggle.textContent = state.avoidance_enabled ? 'Disable obstacle avoidance' : 'Enable obstacle avoidance';
    toggle.setAttribute('aria-pressed', String(Boolean(state.avoidance_enabled)));
    toggle.disabled = avoidanceBusy || !Number.isInteger(state.generation);
    document.getElementById('avoidance-status').textContent = state.avoidance_enabled
      ? `Enabled · ${state.avoidance_status || 'Waiting for ultrasound feedback'}`
      : 'Disabled. Enabling avoidance pauses the controller; click Run to resume.';
    const readings = state.connection === 'connected' && Array.isArray(state.ultrasound) ? state.ultrasound : [];
    const groups = groupReadings(readings, mapping);
    const danger = readings.some(reading => classify(reading) === 'danger')
      || sides.some(side => groups[side].level === 'danger');
    const missing = sides.some(side => groups[side].level === 'unknown' || groups[side].partial);
    const symbol = document.getElementById('ultrasound-alert');
    symbol.textContent = danger ? '!' : missing ? '?' : '•';
    symbol.dataset.level = danger ? 'danger' : missing ? 'unknown' : 'clear';
    for (const side of sides) {
      const group = groups[side];
      document.getElementById(`ultrasound-${side}`).dataset.level = group.level;
      document.getElementById(`ultrasound-${side}-value`).textContent = group.label;
      const selector = document.getElementById(`ultrasound-map-${side}`);
      if (document.activeElement !== selector) {
        const names = [...new Set([...readings.map(r => r.topic), ...group.topics])].sort();
        const signature = JSON.stringify([names, group.topics]);
        if (selector.dataset.options !== signature) {
          selector.replaceChildren(...names.map(topic => {
            const option = document.createElement('option');
            option.value = topic; option.textContent = topic; option.selected = group.topics.includes(topic);
            return option;
          }));
          selector.dataset.options = signature;
        }
      }
    }
    const unmapped = readings.filter(r => !sides.some(side => groups[side].topics.includes(r.topic))).length;
    const detail = state.connection !== 'connected' ? 'Connect ROS to read ultrasound sensors.'
      : !readings.length ? 'Waiting for ultrasound range messages. No fresh sensor data.'
      : danger ? 'Obstacle close! Check red sides and any sensors needing placement.'
      : missing ? 'Some sides have missing, invalid or stale readings.' : 'Receiving ultrasound readings.';
    document.getElementById('ultrasound-detail').textContent = detail
      + (unmapped ? ` ${unmapped} sensor(s) need placement below.` : '');
    const summary = sides.map(side => `${side}: ${groups[side].label} (${groups[side].level})`).join(', ');
    document.getElementById('ultrasound-diamond').setAttribute('aria-label', `${danger ? 'Obstacle close. ' : ''}${summary}`);
  }

  for (const side of sides) document.getElementById(`ultrasound-map-${side}`).addEventListener('change', event => {
    mapping[side] = [...event.target.selectedOptions].map(option => option.value);
    try { localStorage.setItem(storageKey, JSON.stringify(mapping)); } catch (_) {}
    const requested = structuredClone(mapping);
    if (lastState.avoidance_enabled) configureAvoidance(true, requested);
    else render(lastState);
  });
  document.getElementById('ultrasound-auto').addEventListener('click', () => {
    mapping = {};
    try { localStorage.removeItem(storageKey); } catch (_) {}
    if (lastState.avoidance_enabled) configureAvoidance(true, {});
    else render(lastState);
  });
  document.getElementById('avoidance-toggle').addEventListener('click', () =>
    configureAvoidance(!lastState.avoidance_enabled, structuredClone(mapping)));
  globalThis.renderUltrasound = render;
})();

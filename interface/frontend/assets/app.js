/* Zero-credential WebSocket Voice & Robotics Client (16 kHz 16-bit Mono PCM) */
(() => {
  const wsBadge = document.getElementById("wsBadge");
  const robotBadge = document.getElementById("robotBadge");
  const estopBtn = document.getElementById("estopBtn");
  const resetEstopBtn = document.getElementById("resetEstopBtn");
  const languageSelect = document.getElementById("languageSelect");
  const dialectSelect = document.getElementById("dialectSelect");
  const rateSlider = document.getElementById("rateSlider");
  const rateLabel = document.getElementById("rateLabel");
  const micToggleBtn = document.getElementById("micToggleBtn");
  const interruptBtn = document.getElementById("interruptBtn");
  const vadMeterFill = document.getElementById("vadMeterFill");
  const partialStatus = document.getElementById("partialStatus");
  const chatTimeline = document.getElementById("chatTimeline");
  const textTurnForm = document.getElementById("textTurnForm");
  const textTurnInput = document.getElementById("textTurnInput");
  const er2Form = document.getElementById("er2Form");
  const er2ImageInput = document.getElementById("er2ImageInput");
  const er2Canvas = document.getElementById("er2Canvas");
  const er2Instruction = document.getElementById("er2Instruction");
  const er2Output = document.getElementById("er2Output");
  const captureCameraBtn = document.getElementById("captureCameraBtn");
  const pantherTelemetryBar = document.getElementById("pantherTelemetryBar");
  const confirmationsList = document.getElementById("confirmationsList");
  const residencyFooter = document.getElementById("residencyFooter");

  let ws = null;
  let micStream = null;
  let audioCtx = null;
  let scriptNode = null;
  let isRecording = false;
  const playbackQueue = [];
  let currentAudio = null;

  function appendMessage(role, text, meta = "", gloss = null, robotNote = null) {
    const div = document.createElement("div");
    div.className = `msg ${role === "user" ? "msg-user" : "msg-assistant"}`;
    const metaEl = document.createElement("div");
    metaEl.className = "msg-meta";
    metaEl.textContent = meta;
    const bodyEl = document.createElement("div");
    bodyEl.textContent = text;
    div.appendChild(metaEl);
    div.appendChild(bodyEl);
    if (gloss) {
      const glossEl = document.createElement("div");
      glossEl.className = "msg-gloss";
      glossEl.textContent = `Standard German gloss: ${gloss}`;
      div.appendChild(glossEl);
    }
    if (robotNote) {
      const rEl = document.createElement("div");
      rEl.className = "msg-gloss";
      rEl.textContent = robotNote;
      div.appendChild(rEl);
    }
    chatTimeline.appendChild(div);
    chatTimeline.scrollTop = chatTimeline.scrollHeight;
  }

  function stopLocalPlayback() {
    playbackQueue.length = 0;
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
  }

  function enqueueWavBase64(b64) {
    if (!b64) return;
    playbackQueue.push(`data:audio/wav;base64,${b64}`);
    playNextInQueue();
  }

  function playNextInQueue() {
    if (currentAudio || playbackQueue.length === 0) return;
    const src = playbackQueue.shift();
    const audio = new Audio(src);
    currentAudio = audio;
    audio.onended = () => {
      currentAudio = null;
      playNextInQueue();
    };
    audio.onerror = () => {
      currentAudio = null;
      playNextInQueue();
    };
    audio.play().catch(() => {
      currentAudio = null;
    });
  }

  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const params = new URLSearchParams({
      language: languageSelect.value,
      dialect: dialectSelect.value,
    });
    ws = new WebSocket(`${proto}//${window.location.host}/ws/voice?${params.toString()}`);

    ws.onopen = () => {
      wsBadge.textContent = "WS: Connected";
      wsBadge.className = "badge badge-ok";
    };

    ws.onclose = () => {
      wsBadge.textContent = "WS: Reconnecting...";
      wsBadge.className = "badge badge-neutral";
      setTimeout(connectWebSocket, 2500);
    };

    function formatRobotStepSummary(c, idx = null) {
      const details = [];
      if (c.target) {
        const zVal = Number(c.target.z || 0);
        const zPart = Math.abs(zVal) > 0.005 ? `, z=${zVal.toFixed(2)}m` : "";
        details.push(`x=${Number(c.target.x).toFixed(2)}m, y=${Number(c.target.y).toFixed(2)}m${zPart}`);
      }
      if (c.yaw_rad !== null && c.yaw_rad !== undefined) {
        const deg = Math.round((Number(c.yaw_rad) * 180) / Math.PI);
        details.push(`yaw=${deg}°`);
      }
      if (c.object_label) {
        details.push(`object="${c.object_label}"`);
      }
      const prefix = idx !== null ? `${idx + 1}. ` : "";
      const detailStr = details.length > 0 ? ` (${details.join(", ")})` : "";
      return `${prefix}${c.action}${detailStr}`;
    }

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "vad_state") {
        partialStatus.textContent = data.state === "speech_started" ? "Listening..." : "Processing utterance...";
        if (data.state === "speech_started") {
          stopLocalPlayback();
        }
      } else if (data.type === "partial_transcript") {
        partialStatus.textContent = `Partial (${data.language}): ${data.text}`;
      } else if (data.type === "final_transcript") {
        const t = data.transcription;
        const dialectTag = t.detected_dialect ? ` · ${t.detected_dialect}` : "";
        partialStatus.textContent =
          languageSelect.value === "auto" ? `Auto-detected: ${t.language}${dialectTag}` : "Idle";
        if (t.text) {
          appendMessage("user", t.text, `You (${t.language}${dialectTag} · ${t.provider})`, t.normalized_text);
        }
      } else if (data.type === "quick_ack") {
        const dialectTag = data.dialect ? ` · ${data.dialect}` : "";
        partialStatus.textContent = "Planning embodied task with Gemini Robotics ER 2...";
        const taskNote = data.translated_task
          ? `Translated for Gemini Robotics ER 2: "${data.translated_task}"`
          : null;
        appendMessage(
          "assistant",
          data.text,
          `Assistant (${data.language}${dialectTag} · ${data.model || "gemini-flash"})`,
          null,
          taskNote
        );
      } else if (data.type === "assistant_response") {
        const a = data.assistant;
        const dialectTag = a.dialect ? ` · ${a.dialect}` : "";
        partialStatus.textContent =
          languageSelect.value === "auto" ? `Auto-detected: ${a.language}${dialectTag}` : "Idle";
        let robotNote = null;
        if (a.robot_commands && a.robot_commands.length > 0) {
          const multi = a.robot_commands.length > 1;
          const summaries = a.robot_commands.map((c, idx) => formatRobotStepSummary(c, multi ? idx : null));
          robotNote = `ER 2 plan queued (${a.robot_commands.length} step${multi ? "s" : ""}): ${summaries.join(" → ")}`;
        } else if (a.robot_rejections && a.robot_rejections.length > 0) {
          robotNote = `Safety gate blocked: ${a.robot_rejections.join("; ")}`;
        }
        appendMessage("assistant", a.text, `Assistant (${a.language}${dialectTag} · ${a.model})`, null, robotNote);
        if (a.camera_frame && a.camera_frame.jpeg_base64) {
          const mime = a.camera_frame.mime_type || "image/jpeg";
          drawDataUrlWithDetections(
            `data:${mime};base64,${a.camera_frame.jpeg_base64}`,
            a.detected_objects || []
          );
          er2Output.innerHTML = `<strong>Auto-Triggered Camera (${a.camera_frame.source || "/camera/color/image_raw"}):</strong> ${a.scene_summary || "Workspace analyzed by Gemini Robotics ER 2."}`;
        }
        if ((a.robot_commands && a.robot_commands.length > 0) || (a.tool_calls && a.tool_calls.length > 0)) {
          refreshRobotStatus();
        }
      } else if (data.type === "tts_audio_chunk") {
        enqueueWavBase64(data.audio_wav_base64);
      } else if (data.type === "interrupted") {
        stopLocalPlayback();
        partialStatus.textContent = "Interrupted";
      } else if (data.type === "estop_latched") {
        stopLocalPlayback();
        updateRobotBadge(data.status);
      } else if (data.type === "error") {
        partialStatus.textContent = `Error: ${data.message}`;
      }
    };
  }

  function syncDialectControlState() {
    const usesSwissDialect = languageSelect.value === "auto" || languageSelect.value === "gsw";
    dialectSelect.disabled = !usesSwissDialect;
    dialectSelect.style.opacity = usesSwissDialect ? "1" : "0.55";
  }

  function pushConfigUpdate() {
    syncDialectControlState();
    rateLabel.textContent = `${Number(rateSlider.value).toFixed(2)}x`;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "config",
          language: languageSelect.value,
          dialect: dialectSelect.value,
          speaking_rate: Number(rateSlider.value),
        })
      );
    }
  }

  languageSelect.addEventListener("change", pushConfigUpdate);
  dialectSelect.addEventListener("change", pushConfigUpdate);
  rateSlider.addEventListener("input", pushConfigUpdate);

  interruptBtn.addEventListener("click", () => {
    stopLocalPlayback();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "interrupt" }));
    }
  });

  function downsampleTo16kPCM(float32Samples, inputSampleRate) {
    const targetRate = 16000;
    if (inputSampleRate === targetRate) {
      const out = new Int16Array(float32Samples.length);
      for (let i = 0; i < float32Samples.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Samples[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return out.buffer;
    }
    const ratio = inputSampleRate / targetRate;
    const outLength = Math.round(float32Samples.length / ratio);
    const result = new Int16Array(outLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < float32Samples.length; i++) {
        accum += float32Samples[i];
        count++;
      }
      const avg = count > 0 ? accum / count : 0;
      const clamped = Math.max(-1, Math.min(1, avg));
      result[offsetResult] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result.buffer;
  }

  async function toggleMicrophone() {
    if (isRecording) {
      isRecording = false;
      micToggleBtn.textContent = "Start Microphone (16 kHz PCM)";
      micToggleBtn.classList.remove("btn-recording");
      vadMeterFill.style.width = "0%";
      if (scriptNode) scriptNode.disconnect();
      if (audioCtx) await audioCtx.close();
      if (micStream) micStream.getTracks().forEach((t) => t.stop());
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "end_of_utterance" }));
      }
      return;
    }

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(micStream);
    scriptNode = audioCtx.createScriptProcessor(2048, 1, 1);

    scriptNode.onaudioprocess = (event) => {
      if (!isRecording) return;
      const input = event.inputBuffer.getChannelData(0);
      let sumSq = 0;
      for (let i = 0; i < input.length; i++) sumSq += input[i] * input[i];
      const rms = Math.sqrt(sumSq / input.length);
      vadMeterFill.style.width = `${Math.min(100, Math.round(rms * 350))}%`;

      if (ws && ws.readyState === WebSocket.OPEN) {
        const pcm16Buf = downsampleTo16kPCM(input, audioCtx.sampleRate);
        ws.send(pcm16Buf);
      }
    };

    source.connect(scriptNode);
    scriptNode.connect(audioCtx.destination);
    isRecording = true;
    micToggleBtn.textContent = "Stop Microphone (Send Utterance)";
    micToggleBtn.classList.add("btn-recording");
  }

  micToggleBtn.addEventListener("click", () => {
    toggleMicrophone().catch((err) => {
      partialStatus.textContent = `Mic error: ${err.message}`;
    });
  });

  textTurnForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const val = textTurnInput.value.trim();
    if (!val) return;
    stopLocalPlayback();
    const langTag = languageSelect.value === "auto" ? "auto-detect" : languageSelect.value;
    appendMessage("user", val, `You (${langTag})`);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "text_input", text: val }));
    }
    textTurnInput.value = "";
  });

  function updateRobotBadge(statusObj) {
    if (!statusObj) return;
    if (statusObj.estop_active || statusObj.state === "estop") {
      robotBadge.textContent = "Robot: E-STOP LATCHED";
      robotBadge.className = "badge badge-danger";
      resetEstopBtn.classList.remove("hidden");
    } else {
      robotBadge.textContent = `Robot: ${statusObj.state.toUpperCase()} (${Math.round(statusObj.battery_percent)}%)`;
      robotBadge.className = "badge badge-ok";
      resetEstopBtn.classList.add("hidden");
    }
    if (pantherTelemetryBar) {
      const linkTxt = statusObj.bridge_connected ? "Connected" : "Standby (No Live Bridge)";
      const camTxt = statusObj.camera_ready
        ? `Ready (${statusObj.camera_source || "/camera/color/image_raw"})`
        : "On-Demand Idle";
      const extras = [];
      if (statusObj.lift_height_m !== null && statusObj.lift_height_m !== undefined) {
        extras.push(`Lift: ${Number(statusObj.lift_height_m).toFixed(2)}m`);
      }
      if (statusObj.front_obstacle_distance_m !== null && statusObj.front_obstacle_distance_m !== undefined) {
        extras.push(`Front LiDAR: ${Number(statusObj.front_obstacle_distance_m).toFixed(2)}m`);
      }
      const extraSuffix = extras.length > 0 ? ` · ${extras.join(" · ")}` : "";
      pantherTelemetryBar.textContent = `Panther ROS 2 (${linkTxt}) · Domain ${statusObj.ros_domain_id || 15} · Camera: ${camTxt}${extraSuffix}`;
    }
  }

  async function refreshRobotStatus() {
    try {
      const res = await fetch("/api/v1/robotics/status");
      if (!res.ok) return;
      const data = await res.json();
      updateRobotBadge(data.status);
      renderConfirmations(data.pending_confirmations || []);
    } catch (_) {}
  }

  function renderConfirmations(pending) {
    confirmationsList.innerHTML = "";
    if (pending.length === 0) {
      const p = document.createElement("p");
      p.className = "muted-text";
      p.textContent = "No pending robot commands.";
      confirmationsList.appendChild(p);
      return;
    }
    if (pending.length > 1) {
      const batchBar = document.createElement("div");
      batchBar.className = "confirm-actions";
      batchBar.style.marginBottom = "0.6rem";
      const confirmAllBtn = document.createElement("button");
      confirmAllBtn.className = "btn btn-primary";
      confirmAllBtn.textContent = `Confirm All ${pending.length} Steps (Sequential)`;
      confirmAllBtn.onclick = async () => {
        confirmAllBtn.disabled = true;
        for (const stepCmd of pending) {
          await fetch(`/api/v1/robotics/confirm/${stepCmd.command_id}`, { method: "POST" });
        }
        refreshRobotStatus();
      };
      batchBar.appendChild(confirmAllBtn);
      confirmationsList.appendChild(batchBar);
    }
    pending.forEach((cmd, idx) => {
      const card = document.createElement("div");
      card.className = "confirm-card";

      const titleEl = document.createElement("strong");
      titleEl.textContent = pending.length > 1 ? `Step ${idx + 1}: ${cmd.action}` : cmd.action;
      card.appendChild(titleEl);

      const paramParts = [];
      if (cmd.target) {
        paramParts.push(
          `target=(${Number(cmd.target.x).toFixed(2)}m, ${Number(cmd.target.y).toFixed(2)}m, ${Number(cmd.target.z).toFixed(2)}m)`
        );
      }
      if (cmd.yaw_rad !== null && cmd.yaw_rad !== undefined) {
        const deg = Math.round((Number(cmd.yaw_rad) * 180) / Math.PI);
        paramParts.push(`yaw=${deg}° (${Number(cmd.yaw_rad).toFixed(2)} rad)`);
      }
      if (cmd.object_label) {
        paramParts.push(`object="${cmd.object_label}"`);
      }
      paramParts.push(`speed=${cmd.speed} m/s`);

      card.appendChild(document.createTextNode(` ${paramParts.join(" · ")}`));
      if (cmd.rationale) {
        card.appendChild(document.createElement("br"));
        const ratEl = document.createElement("span");
        ratEl.className = "muted-text";
        ratEl.textContent = cmd.rationale;
        card.appendChild(ratEl);
      }

      const actions = document.createElement("div");
      actions.className = "confirm-actions";

      const approveBtn = document.createElement("button");
      approveBtn.className = "btn btn-primary";
      approveBtn.textContent = "Confirm & Dispatch";
      approveBtn.onclick = async () => {
        await fetch(`/api/v1/robotics/confirm/${cmd.command_id}`, { method: "POST" });
        refreshRobotStatus();
      };

      const rejectBtn = document.createElement("button");
      rejectBtn.className = "btn btn-secondary";
      rejectBtn.textContent = "Reject";
      rejectBtn.onclick = async () => {
        await fetch(`/api/v1/robotics/reject/${cmd.command_id}`, { method: "POST" });
        refreshRobotStatus();
      };

      actions.appendChild(approveBtn);
      actions.appendChild(rejectBtn);
      card.appendChild(actions);
      confirmationsList.appendChild(card);
    });
  }

  estopBtn.addEventListener("click", async () => {
    stopLocalPlayback();
    await fetch("/api/v1/robotics/estop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason: "Web UI operator E-STOP button pressed" }),
    });
    refreshRobotStatus();
  });

  resetEstopBtn.addEventListener("click", async () => {
    await fetch("/api/v1/robotics/estop/reset", { method: "POST" });
    refreshRobotStatus();
  });

  if (captureCameraBtn) {
    captureCameraBtn.addEventListener("click", async () => {
      er2Output.textContent = "Triggering Panther ROS 2 camera (/camera/color/image_raw)...";
      try {
        const res = await fetch("/api/v1/robotics/camera/capture", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: "Web UI manual camera trigger", max_age_seconds: 0.0 }),
        });
        const data = await res.json();
        if (!res.ok) {
          er2Output.textContent = `Camera trigger: ${data.detail || res.statusText}`;
          return;
        }
        const f = data.frame;
        const mime = f.mime_type || "image/jpeg";
        drawDataUrlWithDetections(`data:${mime};base64,${f.jpeg_base64}`, []);
        const depthNote =
          f.center_depth_m !== null && f.center_depth_m !== undefined
            ? ` · Center depth: ${Number(f.center_depth_m).toFixed(2)}m`
            : "";
        er2Output.innerHTML = `<strong>Captured Panther Camera:</strong> <code>${f.source}</code> (${f.width}×${f.height}${depthNote})`;
        refreshRobotStatus();
      } catch (err) {
        er2Output.textContent = `Camera trigger failed: ${err.message}`;
      }
    });
  }

  er2Form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const hasFile = Boolean(er2ImageInput.files && er2ImageInput.files[0]);
    er2Output.textContent = hasFile
      ? "Running Gemini Robotics ER 2 spatial analysis..."
      : "Triggering Panther camera (/camera/color/image_raw) & running Gemini Robotics ER 2...";
    const fd = new FormData();
    fd.append("instruction", er2Instruction.value);
    if (hasFile) {
      fd.append("image", er2ImageInput.files[0]);
      drawPreviewWithDetections(er2ImageInput.files[0], []);
    }
    try {
      const res = await fetch("/api/v1/robotics/analyze", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        er2Output.textContent = `Error: ${data.detail || res.statusText}`;
        return;
      }
      const r = data.reasoning;
      const camTag =
        data.camera_captured && data.camera_frame
          ? ` · Camera: ${data.camera_frame.source}`
          : "";
      er2Output.innerHTML = `<strong>Scene (${r.model}${camTag}):</strong> ${r.scene_summary}<br/><span class="muted-text">Progress: ${r.task_progress_percent}% · Validated proposals: ${data.validated_commands.length}</span>`;
      if (hasFile) {
        drawPreviewWithDetections(er2ImageInput.files[0], r.detected_objects || []);
      } else if (data.camera_frame && data.camera_frame.jpeg_base64) {
        const mime = data.camera_frame.mime_type || "image/jpeg";
        drawDataUrlWithDetections(
          `data:${mime};base64,${data.camera_frame.jpeg_base64}`,
          r.detected_objects || []
        );
      }
      refreshRobotStatus();
    } catch (err) {
      er2Output.textContent = `Request failed: ${err.message}`;
    }
  });

  function drawDataUrlWithDetections(dataUrl, objects) {
    const img = new Image();
    img.onload = () => {
      er2Canvas.classList.remove("hidden");
      const ctx = er2Canvas.getContext("2d");
      er2Canvas.width = img.width;
      er2Canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      ctx.lineWidth = Math.max(2, Math.round(img.width / 220));
      ctx.font = `${Math.max(14, Math.round(img.width / 32))}px sans-serif`;
      objects.forEach((obj) => {
        ctx.strokeStyle = "#22c55e";
        ctx.fillStyle = "#22c55e";
        if (obj.box_2d_norm) {
          const [ymin, xmin, ymax, xmax] = obj.box_2d_norm;
          const x = (xmin / 1000) * img.width;
          const y = (ymin / 1000) * img.height;
          const w = ((xmax - xmin) / 1000) * img.width;
          const h = ((ymax - ymin) / 1000) * img.height;
          ctx.strokeRect(x, y, w, h);
          ctx.fillText(obj.label, x + 4, Math.max(18, y - 6));
        } else if (obj.point_yx_norm) {
          const [py, px] = obj.point_yx_norm;
          const cx = (px / 1000) * img.width;
          const cy = (py / 1000) * img.height;
          ctx.beginPath();
          ctx.arc(cx, cy, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillText(obj.label, cx + 12, cy + 4);
        }
      });
    };
    img.src = dataUrl;
  }

  function drawPreviewWithDetections(file, objects) {
    const reader = new FileReader();
    reader.onload = (e) => {
      drawDataUrlWithDetections(e.target.result, objects);
    };
    reader.readAsDataURL(file);
  }

  async function loadPublicConfig() {
    try {
      const res = await fetch("/api/v1/config/public");
      if (!res.ok) return;
      const cfg = await res.json();
      const d = cfg.residency_disclosures;
      residencyFooter.textContent =
        `Hosting: ${d.hosting_region} (Zurich) · STT/TTS: ${d.stt_endpoint_region.toUpperCase()} multi-region · ` +
        `Gemini LLM: ${d.llm_endpoint_region} · Robotics ER 2: ${d.robotics_er2_endpoint} · ` +
        `SwissDial TTS license: ${d.piper_swissdial_license}`;
    } catch (_) {}
  }

  syncDialectControlState();
  connectWebSocket();
  loadPublicConfig();
  refreshRobotStatus();
  setInterval(refreshRobotStatus, 5000);
})();

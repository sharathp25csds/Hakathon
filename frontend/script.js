/* =========================================================
   SIGN BRIDGE AI — Frontend logic (vanilla JavaScript, no frameworks)
   ---------------------------------------------------------
   Sections:
     1. Config & settings storage
     2. Helpers (DOM, toasts, activity log)
     3. Navigation (5 sections, hash based)
     4. Backend API layer  <-- connect the Python backend here
     5. Result rendering   <-- customize how AI output is drawn
     6. Dashboard
     7. Text emotion module
     8. Voice emotion module
     9. Facial emotion module
    10. Settings module
    11. Init
   No AI results are generated in the browser. Every result slot
   stays empty until a real backend responds.
   ========================================================= */

(function () {
  "use strict";

  /* ---------- 1. Config & settings storage ---------- */
  var STORAGE_KEY = "emotix.settings";
  var DEFAULTS = {
    backendUrl: "",
    endpoints: {
      health: "/health",
      text: "/api/analyze/text",
      voice: "/api/analyze/voice",
      face: "/api/analyze/face"
    },
    timeoutSec: 30
  };

  function loadSettings() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return clone(DEFAULTS);
      var saved = JSON.parse(raw);
      return {
        backendUrl: saved.backendUrl || "",
        endpoints: Object.assign({}, DEFAULTS.endpoints, saved.endpoints || {}),
        timeoutSec: Number(saved.timeoutSec) || DEFAULTS.timeoutSec
      };
    } catch (e) {
      return clone(DEFAULTS);
    }
  }
  function saveSettings(s) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); return true; }
    catch (e) { return false; }
  }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  var settings = loadSettings();
  var backendState = "offline"; // offline | checking | online | error
  var counts = { text: 0, voice: 0, face: 0 };

  /* ---------- 2. Helpers ---------- */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function toast(message, type) {
    var stack = $("#toastStack");
    var el = document.createElement("div");
    el.className = "toast " + (type || "");
    el.textContent = message;
    stack.appendChild(el);
    setTimeout(function () { el.classList.add("hide"); }, 3200);
    setTimeout(function () { el.remove(); }, 3600);
  }

  function timeNow() {
    var d = new Date();
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  }
  function pad(n) { return String(n).padStart(2, "0"); }

  function formatBytes(b) {
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(2) + " MB";
  }

  function logActivity(tag, message) {
    var list = $("#activityList");
    var empty = $(".activity-empty", list);
    if (empty) empty.remove();
    var li = document.createElement("li");
    li.innerHTML = '<time></time><span class="tag"></span><span class="msg"></span>';
    li.querySelector("time").textContent = timeNow();
    li.querySelector(".tag").textContent = tag;
    li.querySelector(".msg").textContent = message;
    list.prepend(li);
  }

  /* ---------- 3. Navigation ---------- */
  var SECTIONS = ["dashboard", "text", "voice", "face", "settings"];

  function showSection(name) {
    if (SECTIONS.indexOf(name) === -1) name = "dashboard";
    $$(".section").forEach(function (sec) {
      sec.classList.toggle("active", sec.id === "section-" + name);
    });
    $$(".nav-link").forEach(function (link) {
      var active = link.dataset.section === name;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    var sec = $("#section-" + name);
    $("#pageTitle").textContent = sec.dataset.title;
    $("#pageEyebrow").textContent = sec.dataset.eyebrow;
    document.title = sec.dataset.title + " · SIGN BRIDGE AI";
    closeSidebar();
    window.scrollTo({ top: 0, behavior: "smooth" });

    // Release the camera when leaving the face section.
    if (name !== "face") stopCamera();
  }

  function currentHash() { return (location.hash || "#dashboard").slice(1); }

  function openSidebar() { $("#sidebar").classList.add("open"); $("#sidebarBackdrop").classList.add("show"); }
  function closeSidebar() { $("#sidebar").classList.remove("open"); $("#sidebarBackdrop").classList.remove("show"); }

  function initNavigation() {
    window.addEventListener("hashchange", function () { showSection(currentHash()); });
    $("#menuToggle").addEventListener("click", openSidebar);
    $("#sidebarBackdrop").addEventListener("click", closeSidebar);
    showSection(currentHash());
  }

  /* ---------- 4. Backend API layer ----------
     All network calls live here. The Python backend only needs to
     expose the endpoints configured in Settings and return JSON. */
  var EmotixAPI = {
    isConfigured: function () { return !!settings.backendUrl; },

    url: function (path) {
      return settings.backendUrl.replace(/\/+$/, "") + "/" + String(path || "").replace(/^\/+/, "");
    },

    request: function (path, options) {
      var controller = new AbortController();
      var timer = setTimeout(function () { controller.abort(); }, settings.timeoutSec * 1000);
      var opts = Object.assign({ signal: controller.signal }, options || {});
      return fetch(this.url(path), opts)
        .then(function (res) {
          clearTimeout(timer);
          var type = res.headers.get("content-type") || "";
          var body = type.indexOf("application/json") !== -1 ? res.json() : res.text();
          return body.then(function (data) {
            if (!res.ok) {
              var err = new Error("Server responded with " + res.status);
              err.data = data;
              throw err;
            }
            return data;
          });
        })
        .catch(function (err) {
          clearTimeout(timer);
          if (err.name === "AbortError") throw new Error("Request timed out after " + settings.timeoutSec + "s");
          throw err;
        });
    },

    health: function () { return this.request(settings.endpoints.health, { method: "GET" }); },

    analyzeText: function (payload) {
      return this.request(settings.endpoints.text, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
    },

    analyzeVoice: function (blob, filename) {
      var fd = new FormData();
      fd.append("audio", blob, filename || "recording.webm");
      return this.request(settings.endpoints.voice, { method: "POST", body: fd });
    },

    analyzeFace: function (blob, filename) {
      var fd = new FormData();
      fd.append("image", blob, filename || "capture.png");
      return this.request(settings.endpoints.face, { method: "POST", body: fd });
    }
  };
  window.EmotixAPI = EmotixAPI; // handy for debugging from the browser console

  /* ---------- 5. Result rendering ----------
     Each AI slot has 4 states: empty, loading, error, data.
     renderResult() currently prints the backend JSON as-is.
     Replace its body to draw charts/labels once the response
     format from the Python team is final. */
  var SLOT_EMPTY_HTML = {};

  function rememberSlot(slot) {
    if (!SLOT_EMPTY_HTML[slot.id]) SLOT_EMPTY_HTML[slot.id] = slot.innerHTML;
  }

  function setSlot(slot, state, title, text) {
    rememberSlot(slot);
    slot.classList.remove("is-loading", "is-error", "has-data");
    if (state === "empty") { slot.innerHTML = SLOT_EMPTY_HTML[slot.id]; return; }
    if (state === "loading") slot.classList.add("is-loading");
    if (state === "error") slot.classList.add("is-error");
    slot.innerHTML = "";
    var t = document.createElement("p"); t.className = "ai-slot-title"; t.textContent = title;
    var p = document.createElement("p"); p.className = "ai-slot-text"; p.textContent = text || "";
    slot.appendChild(t); slot.appendChild(p);
  }

  function renderResult(slot, data) {
    rememberSlot(slot);
    slot.classList.remove("is-loading", "is-error");
    slot.classList.add("has-data");
    slot.innerHTML = "";
    var t = document.createElement("p");
    t.className = "ai-slot-title";
    t.textContent = "Backend response";
    var pre = document.createElement("pre");
    pre.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
    slot.appendChild(t);
    slot.appendChild(pre);
  }

  /* Shared submit flow for all three modules. */
  function submitToBackend(kind, slot, call, label) {
    if (!EmotixAPI.isConfigured()) {
      setSlot(slot, "error", "Backend not connected",
        "Add your backend URL in Settings, then try again. Nothing was sent.");
      toast("Set a backend URL in Settings first.", "error");
      logActivity(kind, label + " — not sent (no backend)");
      return Promise.resolve();
    }
    setSlot(slot, "loading", "Sending to backend…", "Waiting for the emotion model to respond.");
    counts[kind]++;
    updateStats();
    logActivity(kind, label + " — sent");
    return call()
      .then(function (data) {
        renderResult(slot, data);
        logActivity(kind, "Response received");
        toast("Analysis received.", "success");
      })
      .catch(function (err) {
        setSlot(slot, "error", "Request failed", err.message || "Could not reach the backend.");
        logActivity(kind, "Request failed: " + (err.message || "error"));
        toast("Backend request failed.", "error");
      });
  }

  /* ---------- 6. Dashboard ---------- */
  function updateStats() {
    $("#statText").textContent = counts.text;
    $("#statVoice").textContent = counts.voice;
    $("#statFace").textContent = counts.face;
  }

  function setBackendState(state, detail) {
    backendState = state;
    var labels = { offline: "Backend not connected", checking: "Checking backend…", online: "Backend online", error: "Backend unreachable" };
    var pill = $("#sidebarStatus");
    pill.dataset.state = state;
    $(".status-text", pill).textContent = labels[state];

    var stat = $("#statBackend");
    stat.textContent = { offline: "Offline", checking: "Checking", online: "Online", error: "Error" }[state];
    stat.classList.toggle("online", state === "online");
    stat.classList.toggle("error", state === "error");
    $("#statBackendUrl").textContent = settings.backendUrl || "No URL set";

    setCheck("backend",
      state === "online" ? "ok" : state === "offline" ? "warn" : state === "checking" ? "" : "fail",
      detail || { offline: "not set", checking: "…", online: "yes", error: "no" }[state]);
  }

  function checkBackend(silent) {
    if (!EmotixAPI.isConfigured()) { setBackendState("offline"); return Promise.resolve(false); }
    setBackendState("checking");
    return EmotixAPI.health()
      .then(function () {
        setBackendState("online");
        if (!silent) toast("Backend is reachable.", "success");
        return true;
      })
      .catch(function (err) {
        setBackendState("error");
        if (!silent) toast("Could not reach backend: " + err.message, "error");
        return false;
      });
  }

  function setCheck(name, state, value) {
    var li = $('#systemChecks li[data-check="' + name + '"]');
    if (!li) return;
    li.classList.remove("ok", "fail", "warn");
    if (state) li.classList.add(state);
    $(".check-val", li).textContent = value;
  }

  function runSystemChecks() {
    var media = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    setCheck("mic", media ? "ok" : "fail", media ? "available" : "unsupported");
    setCheck("cam", media ? "ok" : "fail", media ? "available" : "unsupported");
    var rec = typeof window.MediaRecorder !== "undefined";
    setCheck("recorder", rec ? "ok" : "fail", rec ? "available" : "unsupported");
    var store = true;
    try { localStorage.setItem("emotix.test", "1"); localStorage.removeItem("emotix.test"); } catch (e) { store = false; }
    setCheck("storage", store ? "ok" : "warn", store ? "available" : "blocked");
    if (!window.isSecureContext && media) {
      setCheck("mic", "warn", "needs localhost/https");
      setCheck("cam", "warn", "needs localhost/https");
    }
    checkBackend(true);
  }

  function initDashboard() {
    $("#rerunChecks").addEventListener("click", function () { runSystemChecks(); toast("System check re-run."); });
    $("#clearActivity").addEventListener("click", function () {
      $("#activityList").innerHTML = '<li class="activity-empty">No activity yet. Submit text, audio or an image to see it logged here.</li>';
    });
    runSystemChecks();
  }

  /* ---------- 7. Text emotion ---------- */
  function initText() {
    var input = $("#textInput");
    var count = $("#charCount");
    var slot = $("#textResult");
    rememberSlot(slot);

    input.addEventListener("input", function () { count.textContent = input.value.length; });

    $("#textClear").addEventListener("click", function () {
      input.value = ""; count.textContent = "0"; setSlot(slot, "empty"); input.focus();
    });

    $("#textForm").addEventListener("submit", function (e) {
      e.preventDefault();
      var text = input.value.trim();
      if (!text) { toast("Please enter some text first.", "error"); input.focus(); return; }
      var btn = $("#textSubmit");
      btn.disabled = true;
      var payload = { text: text, language: $("#textLang").value, mode: $("#textMode").value };
      submitToBackend("text", slot, function () { return EmotixAPI.analyzeText(payload); },
        "Text (" + text.length + " chars)")
        .finally(function () { btn.disabled = false; });
    });

    // Ctrl/Cmd + Enter submits.
    input.addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("#textForm").requestSubmit();
    });
  }

  /* ---------- 8. Voice emotion ---------- */
  var voice = { recorder: null, stream: null, chunks: [], blob: null, filename: "", timer: null, start: 0, audioCtx: null, raf: null, url: null };

  function drawIdleWave() {
    var c = $("#waveCanvas"), ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.strokeStyle = "#3A3647"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, c.height / 2); ctx.lineTo(c.width, c.height / 2); ctx.stroke();
  }

  function startVisualizer(stream) {
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    voice.audioCtx = new AC();
    var src = voice.audioCtx.createMediaStreamSource(stream);
    var analyser = voice.audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    var data = new Uint8Array(analyser.fftSize);
    var c = $("#waveCanvas"), ctx = c.getContext("2d");
    var grad = ctx.createLinearGradient(0, 0, c.width, 0);
    grad.addColorStop(0, "#FF7A59"); grad.addColorStop(1, "#FFC24B");
    (function draw() {
      voice.raf = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(data);
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.lineWidth = 2.5; ctx.strokeStyle = grad; ctx.beginPath();
      var step = c.width / data.length;
      for (var i = 0; i < data.length; i++) {
        var y = (data[i] / 255) * c.height;
        if (i === 0) ctx.moveTo(0, y); else ctx.lineTo(i * step, y);
      }
      ctx.stroke();
    })();
  }

  function stopVisualizer() {
    if (voice.raf) cancelAnimationFrame(voice.raf);
    voice.raf = null;
    if (voice.audioCtx) { voice.audioCtx.close(); voice.audioCtx = null; }
    drawIdleWave();
  }

  function setAudio(blob, filename, meta) {
    voice.blob = blob; voice.filename = filename;
    if (voice.url) URL.revokeObjectURL(voice.url);
    voice.url = URL.createObjectURL(blob);
    $("#audioPreview").src = voice.url;
    $("#audioMeta").textContent = meta;
    $("#audioPreviewWrap").hidden = false;
    $("#voiceSubmit").disabled = false;
    $("#voiceReset").disabled = false;
  }

  function resetAudio() {
    voice.blob = null; voice.filename = "";
    if (voice.url) { URL.revokeObjectURL(voice.url); voice.url = null; }
    $("#audioPreview").removeAttribute("src");
    $("#audioPreviewWrap").hidden = true;
    $("#voiceSubmit").disabled = true;
    $("#voiceReset").disabled = true;
    $("#audioFile").value = "";
    $("#recTimer").textContent = "00:00";
    setSlot($("#voiceResult"), "empty");
  }

  function startRecording() {
    if (!navigator.mediaDevices || !window.MediaRecorder) { toast("Recording isn't supported in this browser.", "error"); return; }
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(function (stream) {
        resetAudio();
        voice.stream = stream;
        voice.chunks = [];
        voice.recorder = new MediaRecorder(stream);
        voice.recorder.ondataavailable = function (e) { if (e.data.size) voice.chunks.push(e.data); };
        voice.recorder.onstop = function () {
          var type = voice.recorder.mimeType || "audio/webm";
          var blob = new Blob(voice.chunks, { type: type });
          var ext = type.indexOf("ogg") !== -1 ? "ogg" : type.indexOf("mp4") !== -1 ? "m4a" : "webm";
          var secs = Math.round((Date.now() - voice.start) / 1000);
          setAudio(blob, "recording." + ext, "Recording · " + secs + "s · " + formatBytes(blob.size));
          logActivity("voice", "Recorded " + secs + "s of audio");
        };
        voice.recorder.start();
        voice.start = Date.now();
        voice.timer = setInterval(function () {
          var s = Math.floor((Date.now() - voice.start) / 1000);
          $("#recTimer").textContent = pad(Math.floor(s / 60)) + ":" + pad(s % 60);
        }, 250);
        startVisualizer(stream);
        $("#recStart").disabled = true;
        $("#recStart").classList.add("recording");
        $("#recStart").textContent = "Recording…";
        $("#recStop").disabled = false;
        $("#recIndicator").hidden = false;
      })
      .catch(function () { toast("Microphone permission was denied.", "error"); });
  }

  function stopRecording() {
    if (voice.recorder && voice.recorder.state !== "inactive") voice.recorder.stop();
    if (voice.stream) voice.stream.getTracks().forEach(function (t) { t.stop(); });
    voice.stream = null;
    clearInterval(voice.timer);
    stopVisualizer();
    $("#recStart").disabled = false;
    $("#recStart").classList.remove("recording");
    $("#recStart").textContent = "Start recording";
    $("#recStop").disabled = true;
    $("#recIndicator").hidden = true;
  }

  function handleAudioFile(file) {
    if (!file) return;
    if (file.type && file.type.indexOf("audio/") !== 0) { toast("Please choose an audio file.", "error"); return; }
    setAudio(file, file.name, file.name + " · " + formatBytes(file.size));
    logActivity("voice", "Loaded file " + file.name);
  }

  function initVoice() {
    rememberSlot($("#voiceResult"));
    drawIdleWave();
    $("#recStart").addEventListener("click", startRecording);
    $("#recStop").addEventListener("click", stopRecording);
    $("#audioFile").addEventListener("change", function (e) { handleAudioFile(e.target.files[0]); });
    setupDropzone($("#audioDrop"), handleAudioFile);
    $("#voiceReset").addEventListener("click", resetAudio);
    $("#voiceSubmit").addEventListener("click", function () {
      if (!voice.blob) return;
      var btn = this; btn.disabled = true;
      submitToBackend("voice", $("#voiceResult"),
        function () { return EmotixAPI.analyzeVoice(voice.blob, voice.filename); },
        "Audio " + voice.filename)
        .finally(function () { btn.disabled = !voice.blob; });
    });
  }

  /* ---------- 9. Facial emotion ---------- */
  var face = { stream: null, blob: null, filename: "", url: null };

  function startCamera() {
    if (!navigator.mediaDevices) { toast("Camera isn't supported in this browser.", "error"); return; }
    navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false })
      .then(function (stream) {
        face.stream = stream;
        $("#camVideo").srcObject = stream;
        $("#camEmpty").hidden = true;
        $("#camStart").disabled = true;
        $("#camStop").disabled = false;
        $("#camCapture").disabled = false;
        logActivity("face", "Camera started");
      })
      .catch(function () { toast("Camera permission was denied.", "error"); });
  }

  function stopCamera() {
    if (!face.stream) return;
    face.stream.getTracks().forEach(function (t) { t.stop(); });
    face.stream = null;
    $("#camVideo").srcObject = null;
    $("#camEmpty").hidden = false;
    $("#camStart").disabled = false;
    $("#camStop").disabled = true;
    $("#camCapture").disabled = true;
  }

  function setImage(blob, filename, meta) {
    face.blob = blob; face.filename = filename;
    if (face.url) URL.revokeObjectURL(face.url);
    face.url = URL.createObjectURL(blob);
    $("#imagePreview").src = face.url;
    $("#imageMeta").textContent = meta;
    $("#imagePreviewWrap").hidden = false;
    $("#faceSubmit").disabled = false;
    $("#faceReset").disabled = false;
  }

  function resetImage() {
    face.blob = null; face.filename = "";
    if (face.url) { URL.revokeObjectURL(face.url); face.url = null; }
    $("#imagePreview").removeAttribute("src");
    $("#imagePreviewWrap").hidden = true;
    $("#faceSubmit").disabled = true;
    $("#faceReset").disabled = true;
    $("#imageFile").value = "";
    setSlot($("#faceResult"), "empty");
  }

  function captureFrame() {
    var video = $("#camVideo");
    if (!video.videoWidth) { toast("Camera is still starting…"); return; }
    var canvas = $("#captureCanvas");
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    var ctx = canvas.getContext("2d");
    ctx.translate(canvas.width, 0); ctx.scale(-1, 1); // match mirrored preview
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(function (blob) {
      setImage(blob, "capture.png", "Captured frame · " + canvas.width + "×" + canvas.height + " · " + formatBytes(blob.size));
      logActivity("face", "Captured a frame");
    }, "image/png");
  }

  function handleImageFile(file) {
    if (!file) return;
    if (file.type.indexOf("image/") !== 0) { toast("Please choose an image file.", "error"); return; }
    setImage(file, file.name, file.name + " · " + formatBytes(file.size));
    logActivity("face", "Loaded image " + file.name);
  }

  function initFace() {
    rememberSlot($("#faceResult"));
    $$("[data-face-tab]").forEach(function (tab) {
      tab.addEventListener("click", function () {
        var name = tab.dataset.faceTab;
        $$("[data-face-tab]").forEach(function (t) { t.classList.toggle("active", t === tab); });
        $$("[data-face-panel]").forEach(function (p) { p.hidden = p.dataset.facePanel !== name; });
        if (name !== "camera") stopCamera();
      });
    });
    $("#camStart").addEventListener("click", startCamera);
    $("#camStop").addEventListener("click", stopCamera);
    $("#camCapture").addEventListener("click", captureFrame);
    $("#imageFile").addEventListener("change", function (e) { handleImageFile(e.target.files[0]); });
    setupDropzone($("#imageDrop"), handleImageFile);
    $("#faceReset").addEventListener("click", resetImage);
    $("#faceSubmit").addEventListener("click", function () {
      if (!face.blob) return;
      var btn = this; btn.disabled = true;
      submitToBackend("face", $("#faceResult"),
        function () { return EmotixAPI.analyzeFace(face.blob, face.filename); },
        "Image " + face.filename)
        .finally(function () { btn.disabled = !face.blob; });
    });
  }

  function setupDropzone(zone, onFile) {
    ["dragenter", "dragover"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("dragover"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove("dragover"); });
    });
    zone.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files[0];
      if (f) onFile(f);
    });
  }

  /* ---------- 10. Settings ---------- */
  function fillSettingsForm() {
    $("#backendUrl").value = settings.backendUrl;
    $("#epHealth").value = settings.endpoints.health;
    $("#epText").value = settings.endpoints.text;
    $("#epVoice").value = settings.endpoints.voice;
    $("#epFace").value = settings.endpoints.face;
    $("#timeoutSec").value = settings.timeoutSec;
  }

  function readSettingsForm() {
    var url = $("#backendUrl").value.trim();
    if (url && !/^https?:\/\//i.test(url)) url = "http://" + url;
    var t = parseInt($("#timeoutSec").value, 10);
    return {
      backendUrl: url,
      endpoints: {
        health: $("#epHealth").value.trim() || DEFAULTS.endpoints.health,
        text: $("#epText").value.trim() || DEFAULTS.endpoints.text,
        voice: $("#epVoice").value.trim() || DEFAULTS.endpoints.voice,
        face: $("#epFace").value.trim() || DEFAULTS.endpoints.face
      },
      timeoutSec: isNaN(t) ? DEFAULTS.timeoutSec : Math.min(300, Math.max(5, t))
    };
  }

  function initSettings() {
    fillSettingsForm();
    $("#settingsForm").addEventListener("submit", function (e) {
      e.preventDefault();
      settings = readSettingsForm();
      fillSettingsForm();
      if (saveSettings(settings)) toast("Settings saved.", "success");
      else toast("Settings applied for this session only (storage blocked).");
      logActivity("system", settings.backendUrl ? "Backend set to " + settings.backendUrl : "Backend URL cleared");
      checkBackend(true);
    });
    $("#testConnection").addEventListener("click", function () {
      settings = readSettingsForm();
      if (!settings.backendUrl) { toast("Enter a backend URL first.", "error"); setBackendState("offline"); return; }
      checkBackend(false);
    });
    $("#resetSettings").addEventListener("click", function () {
      settings = clone(DEFAULTS);
      saveSettings(settings);
      fillSettingsForm();
      setBackendState("offline");
      toast("Settings reset to defaults.");
    });
  }

  /* ---------- 11. Init ---------- */
  function startClock() {
    var el = $("#clock");
    function tick() {
      var d = new Date();
      el.textContent = d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }) + " · " + pad(d.getHours()) + ":" + pad(d.getMinutes());
    }
    tick(); setInterval(tick, 15000);
  }

  document.addEventListener("DOMContentLoaded", function () {
    initText();
    initVoice();
    initFace();
    initSettings();
    initDashboard();
    initNavigation();
    startClock();
    updateStats();
  });
})();

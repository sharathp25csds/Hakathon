/* =========================================================
   SIGN BRIDGE AI — Assistive ISL & Communication Platform
   ---------------------------------------------------------
   Modules:
     1. Settings & Config
     2. Core Helpers (Toasts, Activity Log, Diagnostics)
     3. Navigation (Dashboard, Sign, TTS, Face, History, Settings)
     4. Backend API Layer
     5. Sign Language Recognition (Webcam, Upload, Letter/Word/Sentence Buffer)
     6. Text-to-Speech (Two-Way Communication, Browser SpeechSynthesis)
     7. Facial Expression Analysis (Camera, Contextual Sentiment, Standby)
     8. Communication History (Dialogue Log, Filtering, Re-speak)
     9. Diagnostics & Initialization
   ========================================================= */

(function () {
  "use strict";

  /* ---------- 1. Settings & Config ---------- */
  var STORAGE_SETTINGS = "signbridge.settings";
  var STORAGE_HISTORY = "signbridge.history";

  function getDefaultBaseUrl() {
    if (window.location.protocol === "http:" || window.location.protocol === "https:") {
      return window.location.origin;
    }
    return "http://localhost:8000";
  }

  var DEFAULTS = {
    backendUrl: getDefaultBaseUrl(),
    endpoints: {
      health: "/api/health",
      predict: "/api/predict",
      supported: "/api/supported-gestures",
      sentenceClear: "/api/sentence/clear",
      sentenceBackspace: "/api/sentence/backspace",
      sentenceSpace: "/api/sentence/space",
      sentenceAdd: "/api/sentence/add",
      face: "/api/analyze/face",
      faceStatus: "/api/facial-analysis/status"
    },
    timeoutSec: 15,
    speechRate: 1.0,
    speechPitch: 1.0,
    speechVoice: ""
  };

  function loadSettings() {
    try {
      var raw = localStorage.getItem(STORAGE_SETTINGS);
      if (!raw) return Object.assign({}, DEFAULTS);
      var saved = JSON.parse(raw);
      return {
        backendUrl: saved.backendUrl || getDefaultBaseUrl(),
        endpoints: Object.assign({}, DEFAULTS.endpoints, saved.endpoints || {}),
        timeoutSec: Number(saved.timeoutSec) || DEFAULTS.timeoutSec,
        speechRate: Number(saved.speechRate) || DEFAULTS.speechRate,
        speechPitch: Number(saved.speechPitch) || DEFAULTS.speechPitch,
        speechVoice: saved.speechVoice || ""
      };
    } catch (e) {
      return Object.assign({}, DEFAULTS);
    }
  }

  function saveSettings(s) {
    try {
      localStorage.setItem(STORAGE_SETTINGS, JSON.stringify(s));
      return true;
    } catch (e) {
      return false;
    }
  }

  var settings = loadSettings();
  var sessionStats = {
    signs: 0,
    words: 0,
    spoken: 0,
    backendOnline: false
  };

  /* ---------- 2. Helpers ---------- */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function toast(message, type) {
    var stack = $("#toastStack");
    if (!stack) return;
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
    if (!list) return;
    var empty = $(".activity-empty", list);
    if (empty) empty.remove();
    var li = document.createElement("li");
    li.innerHTML = '<time></time><span class="tag"></span><span class="msg"></span>';
    li.querySelector("time").textContent = timeNow();
    li.querySelector(".tag").textContent = tag;
    li.querySelector(".msg").textContent = message;
    list.prepend(li);
  }

  function updateStatsDisplay() {
    var statSigns = $("#statSigns");
    var statWords = $("#statWords");
    var statSpoken = $("#statSpoken");
    var statBackend = $("#statBackend");
    var statBackendNote = $("#statBackendNote");

    if (statSigns) statSigns.textContent = sessionStats.signs;
    if (statWords) statWords.textContent = sessionStats.words;
    if (statSpoken) statSpoken.textContent = sessionStats.spoken;

    if (statBackend) {
      if (sessionStats.backendOnline) {
        statBackend.textContent = "Online";
        statBackend.style.color = "var(--success)";
        if (statBackendNote) statBackendNote.textContent = "MediaPipe Active";
      } else {
        statBackend.textContent = "Offline";
        statBackend.style.color = "var(--danger)";
        if (statBackendNote) statBackendNote.textContent = "Server Unreachable";
      }
    }
  }

  /* ---------- 3. Navigation ---------- */
  var SECTIONS = ["dashboard", "sign", "tts", "face", "history", "settings"];

  function currentHash() {
    var h = (window.location.hash || "").replace(/^#/, "");
    return SECTIONS.indexOf(h) !== -1 ? h : "dashboard";
  }

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
    if (sec) {
      $("#pageTitle").textContent = sec.dataset.title;
      $("#pageEyebrow").textContent = sec.dataset.eyebrow;
      document.title = sec.dataset.title + " · SIGN BRIDGE AI";
    }
    closeSidebar();
    window.scrollTo({ top: 0, behavior: "smooth" });

    // Stop streams when leaving relevant sections
    if (name !== "sign") stopSignCamera();
    if (name !== "face") stopFaceCamera();
  }

  function closeSidebar() {
    var sidebar = $("#sidebar");
    var backdrop = $("#sidebarBackdrop");
    if (sidebar) sidebar.classList.remove("open");
    if (backdrop) backdrop.classList.remove("show");
  }
  function openSidebar() {
    var sidebar = $("#sidebar");
    var backdrop = $("#sidebarBackdrop");
    if (sidebar) sidebar.classList.add("open");
    if (backdrop) backdrop.classList.add("show");
  }

  function initNavigation() {
    $$(".nav-link").forEach(function (link) {
      link.addEventListener("click", function (e) {
        e.preventDefault();
        var sec = link.dataset.section;
        history.pushState(null, "", "#" + sec);
        showSection(sec);
      });
    });
    window.addEventListener("popstate", function () {
      showSection(currentHash());
    });
    var menuBtn = $("#menuToggle");
    if (menuBtn) menuBtn.addEventListener("click", openSidebar);
    var backdrop = $("#sidebarBackdrop");
    if (backdrop) backdrop.addEventListener("click", closeSidebar);

    showSection(currentHash());
  }

  /* ---------- 4. Backend API Layer ---------- */
  function getApiUrl(path) {
    var base = (settings.backendUrl || "").trim().replace(/\/+$/, "");
    if (!base) return path;
    return base + (path.charAt(0) === "/" ? "" : "/") + path;
  }

  function apiFetch(path, options) {
    options = options || {};
    var timeout = (settings.timeoutSec || 15) * 1000;
    var controller = new AbortController();
    var id = setTimeout(function () { controller.abort(); }, timeout);
    options.signal = controller.signal;

    var url = getApiUrl(path);
    return fetch(url, options)
      .then(function (res) {
        clearTimeout(id);
        if (!res.ok) {
          return res.json().catch(function () {
            throw new Error("HTTP " + res.status + " " + res.statusText);
          }).then(function (err) {
            throw new Error(err.detail || err.error || ("HTTP " + res.status));
          });
        }
        return res.json();
      })
      .catch(function (err) {
        clearTimeout(id);
        if (err.name === "AbortError") throw new Error("Request timed out after " + settings.timeoutSec + "s");
        throw err;
      });
  }

  var currentSignLanguage = "ISL";

  var SignBridgeAPI = {
    checkHealth: function () {
      return apiFetch(settings.endpoints.health);
    },
    getLanguage: function () {
      return apiFetch("/api/language");
    },
    setLanguage: function (language) {
      return apiFetch("/api/language", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: language })
      });
    },
    getSupportedGestures: function (language) {
      var lang = language || currentSignLanguage;
      var path = "/api/supported-gestures?lang=" + encodeURIComponent(lang);
      return apiFetch(path);
    },
    predictSign: function (base64Image, annotate, language) {
      var lang = language || currentSignLanguage;
      return apiFetch(settings.endpoints.predict, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64Image,
          annotate: Boolean(annotate),
          language: lang
        })
      });
    },
    clearSentence: function () {
      return apiFetch(settings.endpoints.sentenceClear, { method: "POST" });
    },
    backspaceSentence: function () {
      return apiFetch(settings.endpoints.sentenceBackspace, { method: "POST" });
    },
    spaceSentence: function () {
      return apiFetch(settings.endpoints.sentenceSpace, { method: "POST" });
    },
    analyzeFace: function (base64Image) {
      return apiFetch(settings.endpoints.face, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image })
      });
    },
    getFacialStatus: function () {
      return apiFetch(settings.endpoints.faceStatus);
    }
  };

  function checkBackendConnection(notify) {
    var pill = $("#sidebarStatus");
    if (pill) {
      pill.dataset.state = "checking";
      $(".status-text", pill).textContent = "Checking backend...";
    }
    return SignBridgeAPI.checkHealth()
      .then(function (data) {
        sessionStats.backendOnline = true;
        if (pill) {
          pill.dataset.state = "online";
          $(".status-text", pill).textContent = "SignBridge AI Online";
        }
        updateStatsDisplay();
        if (notify) toast("Connected to SignBridge AI (" + data.service + ")", "success");
        return data;
      })
      .catch(function (err) {
        sessionStats.backendOnline = false;
        if (pill) {
          pill.dataset.state = "offline";
          $(".status-text", pill).textContent = "Backend Offline";
        }
        updateStatsDisplay();
        if (notify) toast("Backend unreachable: " + err.message, "error");
        throw err;
      });
  }

  /* ---------- 5. Communication History ---------- */
  function loadHistory() {
    try {
      var raw = localStorage.getItem(STORAGE_HISTORY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }
  function saveHistory(hist) {
    try {
      localStorage.setItem(STORAGE_HISTORY, JSON.stringify(hist.slice(0, 50)));
    } catch (e) {}
  }

  var historyLog = loadHistory();
  var currentHistoryFilter = "all";

  function addHistoryEntry(role, text) {
    if (!text || !text.trim()) return;
    var entry = {
      id: Date.now(),
      time: timeNow(),
      role: role, // 'signed' | 'spoken'
      text: text.trim()
    };
    historyLog.unshift(entry);
    saveHistory(historyLog);
    renderHistory();
    logActivity(role === "signed" ? "sign" : "tts", text.trim());
  }

  function renderHistory() {
    var container = $("#historyContainer");
    if (!container) return;
    container.innerHTML = "";

    var filtered = historyLog.filter(function (entry) {
      if (currentHistoryFilter === "all") return true;
      return entry.role === currentHistoryFilter;
    });

    if (filtered.length === 0) {
      var empty = document.createElement("div");
      empty.className = "activity-empty";
      empty.style.cssText = "padding: 30px; text-align: center; border: 1px dashed var(--border); border-radius: var(--radius-sm);";
      empty.textContent = "No communication logged yet for this filter.";
      container.appendChild(empty);
      return;
    }

    filtered.forEach(function (item) {
      var el = document.createElement("div");
      el.className = "history-item";
      var roleClass = item.role === "signed" ? "signed" : "spoken";
      var roleLabel = item.role === "signed" ? "Sign-to-Text" : "Text-to-Speech";

      el.innerHTML =
        '<div class="history-item-head">' +
          '<span class="history-role-tag ' + roleClass + '">' + roleLabel + '</span>' +
          '<span>' + item.time + '</span>' +
        '</div>' +
        '<div style="font-size: 15px; font-weight: 500; color: var(--text);">' + escapeHtml(item.text) + '</div>' +
        '<div class="btn-row" style="margin-top: 4px;">' +
          '<button class="btn btn-small btn-ghost hist-speak-btn" data-text="' + escapeAttr(item.text) + '">Speak Aloud</button>' +
          '<button class="btn btn-small btn-ghost hist-copy-btn" data-text="' + escapeAttr(item.text) + '">Copy</button>' +
        '</div>';

      container.appendChild(el);
    });

    $$(".hist-speak-btn", container).forEach(function (btn) {
      btn.addEventListener("click", function () {
        speakText(btn.dataset.text);
      });
    });
    $$(".hist-copy-btn", container).forEach(function (btn) {
      btn.addEventListener("click", function () {
        navigator.clipboard.writeText(btn.dataset.text).then(function () {
          toast("Copied to clipboard", "success");
        });
      });
    });
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function escapeAttr(str) {
    return str.replace(/"/g, "&quot;");
  }

  function initHistory() {
    renderHistory();
    var clearBtn = $("#clearHistoryBtn");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        historyLog = [];
        saveHistory(historyLog);
        renderHistory();
        toast("History cleared");
      });
    }
    var fAll = $("#filterHistoryAll");
    var fSign = $("#filterHistorySign");
    var fSpeech = $("#filterHistorySpeech");

    if (fAll) fAll.addEventListener("click", function () { currentHistoryFilter = "all"; renderHistory(); });
    if (fSign) fSign.addEventListener("click", function () { currentHistoryFilter = "signed"; renderHistory(); });
    if (fSpeech) fSpeech.addEventListener("click", function () { currentHistoryFilter = "spoken"; renderHistory(); });
  }

  /* ---------- 6. Text-to-Speech (TTS Engine) ---------- */
  var speechSynth = window.speechSynthesis;
  var availableVoices = [];

  function populateVoices() {
    if (!speechSynth) return;
    availableVoices = speechSynth.getVoices();
    var select = $("#ttsVoiceSelect");
    if (!select) return;
    select.innerHTML = '<option value="">Default System Voice</option>';
    availableVoices.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v.name;
      opt.textContent = v.name + " (" + v.lang + ")";
      if (settings.speechVoice && v.name === settings.speechVoice) {
        opt.selected = true;
      }
      select.appendChild(opt);
    });
  }

  if (speechSynth && speechSynth.onvoiceschanged !== undefined) {
    speechSynth.onvoiceschanged = populateVoices;
  }

  function speakText(text) {
    if (!text || !text.trim()) {
      toast("No text to speak.", "error");
      return;
    }
    if (!speechSynth) {
      toast("Browser does not support Speech Synthesis.", "error");
      return;
    }

    speechSynth.cancel(); // Stop any pending speech
    var utter = new SpeechSynthesisUtterance(text.trim());
    utter.rate = settings.speechRate || 1.0;
    utter.pitch = settings.speechPitch || 1.0;

    if (settings.speechVoice && availableVoices.length) {
      var chosen = availableVoices.find(function (v) { return v.name === settings.speechVoice; });
      if (chosen) utter.voice = chosen;
    }

    utter.onstart = function () {
      var btn = $("#ttsSpeakBtn");
      if (btn) btn.classList.add("recording");
    };
    utter.onend = function () {
      var btn = $("#ttsSpeakBtn");
      if (btn) btn.classList.remove("recording");
    };
    utter.onerror = function (e) {
      var btn = $("#ttsSpeakBtn");
      if (btn) btn.classList.remove("recording");
      toast("Speech error: " + e.error, "error");
    };

    speechSynth.speak(utter);
    sessionStats.spoken += 1;
    updateStatsDisplay();
  }

  function initTTS() {
    populateVoices();
    var input = $("#ttsInput");
    var speakBtn = $("#ttsSpeakBtn");
    var stopBtn = $("#ttsStopBtn");
    var clearBtn = $("#ttsClearBtn");
    var rateInput = $("#ttsRate");
    var pitchInput = $("#ttsPitch");
    var voiceSelect = $("#ttsVoiceSelect");
    var charCount = $("#ttsCharCount");

    if (input && charCount) {
      input.addEventListener("input", function () {
        charCount.textContent = input.value.length;
      });
    }

    if (rateInput) {
      rateInput.value = settings.speechRate;
      $("#ttsRateVal").textContent = settings.speechRate.toFixed(1) + "x";
      rateInput.addEventListener("input", function () {
        settings.speechRate = parseFloat(rateInput.value);
        $("#ttsRateVal").textContent = settings.speechRate.toFixed(1) + "x";
        saveSettings(settings);
      });
    }

    if (pitchInput) {
      pitchInput.value = settings.speechPitch;
      $("#ttsPitchVal").textContent = settings.speechPitch.toFixed(1);
      pitchInput.addEventListener("input", function () {
        settings.speechPitch = parseFloat(pitchInput.value);
        $("#ttsPitchVal").textContent = settings.speechPitch.toFixed(1);
        saveSettings(settings);
      });
    }

    if (voiceSelect) {
      voiceSelect.addEventListener("change", function () {
        settings.speechVoice = voiceSelect.value;
        saveSettings(settings);
      });
    }

    if (speakBtn) {
      speakBtn.addEventListener("click", function () {
        var text = input ? input.value : "";
        if (!text.trim()) { toast("Please type a message first.", "error"); return; }
        speakText(text);
        addHistoryEntry("spoken", text);
      });
    }

    if (stopBtn) {
      stopBtn.addEventListener("click", function () {
        if (speechSynth) speechSynth.cancel();
        var btn = $("#ttsSpeakBtn");
        if (btn) btn.classList.remove("recording");
        toast("Speech stopped");
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        if (input) input.value = "";
        if (charCount) charCount.textContent = "0";
      });
    }

    $$(".phrase-chip", $("#section-tts")).forEach(function (chip) {
      chip.addEventListener("click", function () {
        var p = chip.dataset.phrase;
        if (input) {
          input.value = p;
          if (charCount) charCount.textContent = p.length;
        }
        speakText(p);
        addHistoryEntry("spoken", p);
      });
    });
  }

  /* ---------- 7. Sign Recognition Module ---------- */
  var signMedia = {
    stream: null,
    video: null,
    canvas: null,
    isLooping: false,
    loopIntervalId: null,
    uploadFile: null
  };

  function updateSignHUD(data) {
    var letterBadge = $("#signGestureResult");
    var confBadge = $("#signConfidenceResult");
    var confBar = $("#signConfidenceBar");
    var wordBadge = $("#signCurrentWord");
    var sentenceBox = $("#signSentence");
    var statusText = $("#signResultStatus");
    var statusBadge = $("#signStatusBadge");
    var numHands = $("#signNumHands");
    var handedness = $("#signHandedness");
    var latency = $("#signLatency");

    var letter = data.current_letter || data.recognized_text || "—";
    var conf = Math.round((data.confidence || 0) * 100);

    if (letterBadge) letterBadge.textContent = letter;
    if (confBadge) confBadge.textContent = conf + "%";
    if (confBar) confBar.style.width = conf + "%";

    if (wordBadge) wordBadge.textContent = data.current_word || "—";

    if (sentenceBox) {
      if (data.sentence && data.sentence.trim()) {
        sentenceBox.textContent = data.sentence;
        sentenceBox.classList.remove("empty");
      } else {
        sentenceBox.textContent = "No words accumulated yet. Spell letters above to form words.";
        sentenceBox.classList.add("empty");
      }
    }

    if (statusText) statusText.textContent = data.system_status || data.status || "Ready";
    if (statusBadge) statusBadge.textContent = data.hand_detected ? (letter !== "—" ? "Recognized" : "Hand Detected") : "Searching";
    if (numHands) numHands.textContent = data.num_hands || (data.hand_detected ? 1 : 0);
    if (handedness) handedness.textContent = data.handedness || "None";
    if (latency) latency.textContent = (data.latency_ms || 0) + " ms";

    if (letter && letter !== "—") {
      sessionStats.signs += 1;
      updateStatsDisplay();
    }
  }

  function captureFrameBase64(video, canvas) {
    if (!video || !video.videoWidth || !video.videoHeight) return null;
    canvas.width = Math.min(video.videoWidth, 640);
    canvas.height = Math.min(video.videoHeight, 480);
    var ctx = canvas.getContext("2d");
    // Draw mirrored to match user perspective
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  }

  function runPredictionStep() {
    if (!signMedia.stream || !signMedia.video) return;
    var base64 = captureFrameBase64(signMedia.video, signMedia.canvas);
    if (!base64) return;

    SignBridgeAPI.predictSign(base64, false)
      .then(function (res) {
        updateSignHUD(res);
      })
      .catch(function (err) {
        var statusText = $("#signResultStatus");
        if (statusText) statusText.textContent = "Prediction error: " + err.message;
      });
  }

  function startSignLoop() {
    if (signMedia.loopIntervalId) clearInterval(signMedia.loopIntervalId);
    signMedia.isLooping = true;
    signMedia.loopIntervalId = setInterval(function () {
      var toggle = $("#signContinuousToggle");
      if (toggle && toggle.checked && signMedia.stream) {
        runPredictionStep();
      }
    }, 280); // ~3.5 FPS loop gives real-time responsiveness without network overload
  }

  function stopSignLoop() {
    if (signMedia.loopIntervalId) {
      clearInterval(signMedia.loopIntervalId);
      signMedia.loopIntervalId = null;
    }
    signMedia.isLooping = false;
  }

  function startSignCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("Webcam is not supported in this browser.", "error");
      return;
    }

    var startBtn = $("#signCameraStart");
    var stopBtn = $("#signCameraStop");
    var recognizeBtn = $("#signRecognize");
    var statusText = $("#signResultStatus");

    if (startBtn) startBtn.disabled = true;
    if (statusText) statusText.textContent = "Starting camera...";

    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    })
      .then(function (stream) {
        signMedia.stream = stream;
        signMedia.video.srcObject = stream;
        $("#signCameraEmpty").hidden = true;
        if (stopBtn) stopBtn.disabled = false;
        if (recognizeBtn) recognizeBtn.disabled = false;
        if (statusText) statusText.textContent = "Live camera active. Detecting ISL hand signs...";
        toast("Camera started", "success");
        startSignLoop();
      })
      .catch(function (err) {
        if (startBtn) startBtn.disabled = false;
        if (statusText) statusText.textContent = "Camera access denied or unavailable: " + err.message;
        toast("Camera error: " + err.message, "error");
      });
  }

  function stopSignCamera() {
    stopSignLoop();
    if (signMedia.stream) {
      signMedia.stream.getTracks().forEach(function (t) { t.stop(); });
      signMedia.stream = null;
    }
    if (signMedia.video) signMedia.video.srcObject = null;
    var empty = $("#signCameraEmpty");
    if (empty) empty.hidden = false;
    var startBtn = $("#signCameraStart");
    var stopBtn = $("#signCameraStop");
    var recognizeBtn = $("#signRecognize");

    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (recognizeBtn) recognizeBtn.disabled = true;
  }

  function loadSupportedAlphabet(language) {
    var lang = (language || currentSignLanguage || "ISL").toUpperCase();
    var grid = $("#alphabetGrid");
    var badgeTotal = $("#badgeTotalLetters");
    var badgeExp = $("#badgeExperimentalLetters");
    var badgeUnsupp = $("#badgeUnsupportedLetters");
    var badgeLegacy = $("#supportedCountBadge");
    var titleEl = $("#signLanguageTitle");
    var subtitleEl = $("#signLanguageSubtitle");
    var gridTitleEl = $("#alphabetGridTitle");
    var gridSubtitleEl = $("#alphabetGridSubtitle");
    var disclaimerEl = $("#alphabetDisclaimerText");

    if (lang === "ASL") {
      if (titleEl) titleEl.textContent = "American Sign Language (ASL) Fingerspelling Recognition";
      if (subtitleEl) subtitleEl.textContent = "Single-hand fingerspelling recognition. Hold hand steady to assemble words and sentences.";
      if (gridTitleEl) gridTitleEl.textContent = "ASL Alphabet Recognition Status (A–Z)";
      if (gridSubtitleEl) gridSubtitleEl.textContent = "American manual alphabet fingerspelling recognition status";
    } else {
      if (titleEl) titleEl.textContent = "Indian Sign Language (ISL) Fingerspelling Recognition";
      if (subtitleEl) subtitleEl.textContent = "Present one-hand or two-hand alphabet signs to the camera to assemble words and sentences.";
      if (gridTitleEl) gridTitleEl.textContent = "ISL Alphabet Recognition Status (A–Z)";
      if (gridSubtitleEl) gridSubtitleEl.textContent = "Official ISLRTC fingerspelling recognition status";
    }

    if (!grid) return;

    SignBridgeAPI.getSupportedGestures(lang)
      .then(function (data) {
        grid.innerHTML = "";
        var meta = data.metadata || {};
        var counts = data.counts_by_status || {};
        if (disclaimerEl) disclaimerEl.textContent = data.disclaimer || "";

        // Display all 26 letters A through Z in alphabetical order
        var alphabetLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

        var countExperimental = 0;
        var countUnsupported = 0;
        var countTotal = alphabetLetters.length;

        alphabetLetters.forEach(function (letter) {
          var m = meta[letter] || {};
          var status = m.status || "unsupported";

          var card = document.createElement("div");
          var statusClass = "";
          var badgeClass = "";
          var statusLabel = "";

          if (status === "experimental_dynamic") {
            countExperimental += 1;
            statusClass = "status-experimental-dynamic";
            badgeClass = "badge-experimental-dynamic";
            statusLabel = "EXPERIMENTAL · DYNAMIC";
          } else if (status === "experimental") {
            countExperimental += 1;
            statusClass = "status-experimental";
            badgeClass = "badge-experimental";
            statusLabel = "EXPERIMENTAL";
          } else {
            countUnsupported += 1;
            statusClass = "status-unsupported";
            badgeClass = "badge-unsupported";
            statusLabel = "UNSUPPORTED";
          }

          card.className = "alphabet-card " + statusClass;
          card.dataset.letter = letter;
          card.dataset.status = status;

          // Tooltip description from metadata
          var tooltip = letter + ": " + statusLabel;
          if (m.dominant_hand) tooltip += " · " + m.dominant_hand;
          if (m.hands_required) tooltip += " (" + m.hands_required + " hand" + (m.hands_required > 1 ? "s" : "") + ")";
          card.title = tooltip;

          card.innerHTML =
            '<span class="letter">' + letter + '</span>' +
            '<span class="alphabet-status-badge ' + badgeClass + '">' + statusLabel + '</span>';

          grid.appendChild(card);
        });

        // Update count displays
        if (badgeTotal) badgeTotal.textContent = countTotal + " Total";
        if (badgeExp) badgeExp.textContent = countExperimental + " Supported";
        if (badgeUnsupp) badgeUnsupp.textContent = countUnsupported + " Unsupported";
        if (badgeLegacy) {
          badgeLegacy.textContent = countTotal + " Letters (" + countExperimental + " Supported, " + countUnsupported + " Unsupported)";
        }
      })
      .catch(function (err) {
        if (grid) {
          grid.innerHTML =
            '<div class="activity-empty" style="grid-column: 1 / -1; padding: 20px; text-align: center; border: 1px dashed var(--border); border-radius: var(--radius-sm); color: var(--text-dim);">' +
              'Unable to load ' + lang + ' alphabet status from backend (' + escapeHtml(err.message || "Connection offline") + '). Check connection in Settings.' +
            '</div>';
        }
        if (badgeLegacy) badgeLegacy.textContent = "Alphabet unavailable (server offline)";
      });
  }

  function initSign() {
    signMedia.video = $("#signCameraVideo");
    signMedia.canvas = $("#signCaptureCanvas");

    var startBtn = $("#signCameraStart");
    var stopBtn = $("#signCameraStop");
    var recognizeBtn = $("#signRecognize");
    var spaceBtn = $("#signSpaceBtn");
    var backspaceBtn = $("#signBackspaceBtn");
    var clearSentenceBtn = $("#signClearSentenceBtn");
    var speakBtn = $("#signSpeakBtn");

    if (startBtn) startBtn.addEventListener("click", startSignCamera);
    if (stopBtn) stopBtn.addEventListener("click", stopSignCamera);
    if (recognizeBtn) recognizeBtn.addEventListener("click", runPredictionStep);

    // Language Switcher Controls
    var btnISL = $("#btnLangISL");
    var btnASL = $("#btnLangASL");

    function setSignLanguageMode(lang) {
      currentSignLanguage = lang;
      if (btnISL) btnISL.classList.toggle("active", lang === "ISL");
      if (btnASL) btnASL.classList.toggle("active", lang === "ASL");

      SignBridgeAPI.setLanguage(lang)
        .then(function () {
          loadSupportedAlphabet(lang);
          toast("Switched to " + (lang === "ISL" ? "Indian Sign Language (ISL)" : "American Sign Language (ASL)"), "info");
        })
        .catch(function (err) {
          toast("Language switch failed: " + err.message, "error");
        });
    }

    if (btnISL) {
      btnISL.addEventListener("click", function () {
        if (currentSignLanguage !== "ISL") setSignLanguageMode("ISL");
      });
    }
    if (btnASL) {
      btnASL.addEventListener("click", function () {
        if (currentSignLanguage !== "ASL") setSignLanguageMode("ASL");
      });
    }

    // Sentence Controls
    if (spaceBtn) {
      spaceBtn.addEventListener("click", function () {
        SignBridgeAPI.spaceSentence().then(function (res) {
          sessionStats.words += 1;
          updateStatsDisplay();
          updateSignHUD(res);
          toast("Word completed [SPACE]", "success");
        });
      });
    }

    if (backspaceBtn) {
      backspaceBtn.addEventListener("click", function () {
        SignBridgeAPI.backspaceSentence().then(function (res) {
          updateSignHUD(res);
          toast("Backspace");
        });
      });
    }

    if (clearSentenceBtn) {
      clearSentenceBtn.addEventListener("click", function () {
        SignBridgeAPI.clearSentence().then(function (res) {
          updateSignHUD(res);
          toast("Sentence cleared");
        });
      });
    }

    if (speakBtn) {
      speakBtn.addEventListener("click", function () {
        var sentence = ($("#signSentence") ? $("#signSentence").textContent : "").trim();
        if (!sentence || sentence.startsWith("No words")) {
          toast("No completed sentence to speak.", "error");
          return;
        }
        speakText(sentence);
        addHistoryEntry("signed", sentence);
        toast("Speaking sentence: " + sentence, "success");
      });
    }

    // Image Upload Handling
    var dropZone = $("#signImageDrop");
    var fileInput = $("#signImageFile");
    var previewWrap = $("#signImagePreviewWrap");
    var previewImg = $("#signImagePreview");
    var metaText = $("#signImageMeta");
    var analyzeUploadBtn = $("#signUploadAnalyze");
    var clearUploadBtn = $("#signUploadClear");

    function handleSignImageFile(file) {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        toast("Please select an image file (JPG, PNG, WEBP).", "error");
        return;
      }
      var reader = new FileReader();
      reader.onload = function (e) {
        signMedia.uploadBase64 = e.target.result;
        if (previewImg) previewImg.src = signMedia.uploadBase64;
        if (metaText) metaText.textContent = file.name + " (" + formatBytes(file.size) + ")";
        if (previewWrap) previewWrap.hidden = false;
        if (analyzeUploadBtn) analyzeUploadBtn.disabled = false;
        if (clearUploadBtn) clearUploadBtn.disabled = false;
        toast("Image loaded: " + file.name);
      };
      reader.readAsDataURL(file);
    }

    if (fileInput) {
      fileInput.addEventListener("change", function (e) {
        handleSignImageFile(e.target.files[0]);
      });
    }
    if (dropZone) {
      ["dragenter", "dragover"].forEach(function (evt) {
        dropZone.addEventListener(evt, function (e) { e.preventDefault(); dropZone.classList.add("dragover"); });
      });
      ["dragleave", "drop"].forEach(function (evt) {
        dropZone.addEventListener(evt, function (e) { e.preventDefault(); dropZone.classList.remove("dragover"); });
      });
      dropZone.addEventListener("drop", function (e) {
        var f = e.dataTransfer && e.dataTransfer.files[0];
        if (f) handleSignImageFile(f);
      });
    }
    if (analyzeUploadBtn) {
      analyzeUploadBtn.addEventListener("click", function () {
        if (!signMedia.uploadBase64) return;
        analyzeUploadBtn.disabled = true;
        SignBridgeAPI.predictSign(signMedia.uploadBase64, false)
          .then(function (res) {
            analyzeUploadBtn.disabled = false;
            updateSignHUD(res);
            toast("Sign analyzed: " + (res.recognized_text || "No sign detected"), res.hand_detected ? "success" : "info");
          })
          .catch(function (err) {
            analyzeUploadBtn.disabled = false;
            toast("Analysis failed: " + err.message, "error");
          });
      });
    }
    if (clearUploadBtn) {
      clearUploadBtn.addEventListener("click", function () {
        signMedia.uploadBase64 = null;
        if (fileInput) fileInput.value = "";
        if (previewImg) previewImg.src = "";
        if (previewWrap) previewWrap.hidden = true;
        if (analyzeUploadBtn) analyzeUploadBtn.disabled = true;
        if (clearUploadBtn) clearUploadBtn.disabled = true;
      });
    }

    loadSupportedAlphabet();
  }

  /* ---------- 8. Facial Expression Module ---------- */
  var faceMedia = {
    stream: null,
    video: null,
    canvas: null,
    overlayCanvas: null,
    loopIntervalId: null,
    isProcessing: false
  };

  var PERSON_PALETTE = [
    { stroke: "#00e5ff", fill: "rgba(0, 229, 255, 0.14)", border: "#00e5ff", textBg: "#0891b2" }, // Person 1: Cyan
    { stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.14)", border: "#a855f7", textBg: "#9333ea" }, // Person 2: Purple
    { stroke: "#22c55e", fill: "rgba(34, 197, 94, 0.14)", border: "#22c55e", textBg: "#16a34a" },  // Person 3: Green
    { stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.14)", border: "#f59e0b", textBg: "#d97706" }, // Person 4: Amber
    { stroke: "#ec4899", fill: "rgba(236, 72, 153, 0.14)", border: "#ec4899", textBg: "#db2777" }, // Person 5: Pink
    { stroke: "#3b82f6", fill: "rgba(59, 130, 246, 0.14)", border: "#3b82f6", textBg: "#2563eb" }  // Person 6: Blue
  ];

  function getPersonTheme(trackId) {
    var idNum = parseInt(trackId, 10) || 1;
    var idx = Math.max(0, (idNum - 1) % PERSON_PALETTE.length);
    return PERSON_PALETTE[idx];
  }

  function drawFaceOverlays(faces, captureW, captureH) {
    if (!faceMedia.overlayCanvas || !faceMedia.video) return;
    var overlay = faceMedia.overlayCanvas;
    var vw = faceMedia.video.clientWidth;
    var vh = faceMedia.video.clientHeight;
    if (!vw || !vh) return;

    if (overlay.width !== vw || overlay.height !== vh) {
      overlay.width = vw;
      overlay.height = vh;
    }

    var ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, vw, vh);

    if (!faces || faces.length === 0) return;

    var scaleX = vw / (captureW || 640);
    var scaleY = vh / (captureH || 480);

    faces.forEach(function (face) {
      var trackId = face.track_id || face.face_id || 1;
      var theme = getPersonTheme(trackId);
      var bx = face.bbox[0] * scaleX;
      var by = face.bbox[1] * scaleY;
      var bw = face.bbox[2] * scaleX;
      var bh = face.bbox[3] * scaleY;

      // Draw bounding box
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = theme.stroke;
      ctx.fillStyle = theme.fill;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(bx, by, bw, bh, 6);
      } else {
        ctx.rect(bx, by, bw, bh);
      }
      ctx.fill();
      ctx.stroke();

      // Corner accent brackets
      var cLen = Math.min(14, bw * 0.25, bh * 0.25);
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      // TL
      ctx.moveTo(bx, by + cLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cLen, by);
      // TR
      ctx.moveTo(bx + bw - cLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cLen);
      // BL
      ctx.moveTo(bx, by + bh - cLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cLen, by + bh);
      // BR
      ctx.moveTo(bx + bw - cLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cLen);
      ctx.stroke();

      // Tracking Pill Label
      var label = "ID #" + trackId + " · " + (face.expression || "neutral").toUpperCase() + " " + Math.round((face.confidence || 0) * 100) + "%";
      ctx.font = "bold 12px sans-serif";
      var m = ctx.measureText(label);
      var p = 6;
      var lW = m.width + p * 2;
      var lH = 20;
      var lY = Math.max(2, by - lH - 4);
      var lX = Math.min(Math.max(2, bx), vw - lW - 2);

      ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(lX, lY, lW, lH, 4);
      } else {
        ctx.rect(lX, lY, lW, lH);
      }
      ctx.fill();
      ctx.strokeStyle = theme.stroke;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, lX + p, lY + 14);
    });
  }

  function updateFaceHUD(res) {
    var expr = $("#faceExpressionResult");
    var conf = $("#faceConfidenceResult");
    var bar = $("#faceConfidenceBar");
    var status = $("#faceStatusDetail");
    var modBadge = $("#faceModuleStatus");
    var peopleBadge = $("#facePeopleCountBadge");
    var multiContainer = $("#faceMultiResultsContainer");
    var chips = $$(".face-category-chip");

    var numFaces = res.num_faces || (res.faces ? res.faces.length : (res.face_detected ? 1 : 0));
    var faces = res.faces || [];

    // 1. Primary face summary
    if (expr) expr.textContent = res.expression ? res.expression.toUpperCase() : "—";
    var c = Math.round((res.confidence || 0) * 100);
    if (conf) conf.textContent = c + "%";
    if (bar) bar.style.width = c + "%";

    // 2. People Counter badge
    if (peopleBadge) {
      if (numFaces === 1) {
        peopleBadge.textContent = "1 Face Detected";
        peopleBadge.style.background = "rgba(0, 229, 255, 0.15)";
        peopleBadge.style.color = "#00e5ff";
      } else if (numFaces > 1) {
        peopleBadge.textContent = numFaces + " Faces Detected";
        peopleBadge.style.background = "rgba(168, 85, 247, 0.20)";
        peopleBadge.style.color = "#c084fc";
      } else {
        peopleBadge.textContent = "0 Faces";
        peopleBadge.style.background = "";
        peopleBadge.style.color = "";
      }
    }

    if (modBadge) {
      modBadge.textContent = numFaces > 0 ? (numFaces > 1 ? "Tracking " + numFaces : "Tracking") : "Standby";
    }

    // 3. Status detail
    if (status) {
      if (numFaces > 1) {
        status.textContent = numFaces + " persons tracked simultaneously · Primary context: " + (res.expression || "neutral");
      } else if (numFaces === 1) {
        status.textContent = "1 person detected · Context: " + (res.expression || "neutral");
      } else if (res.status === "no_face_detected") {
        status.textContent = "No faces detected in frame";
      } else {
        status.textContent = res.status || "Ready";
      }
    }

    // 4. Multi-Person Cards
    if (multiContainer) {
      if (faces.length === 0) {
        multiContainer.innerHTML =
          '<div class="face-empty-note" style="text-align: center; padding: 14px; color: var(--text-dim); font-size: 13px;">' +
            'No faces currently detected. Point camera towards signers.' +
          '</div>';
      } else {
        var html = "";
        faces.forEach(function (f) {
          var trackId = f.track_id || f.face_id || 1;
          var theme = getPersonTheme(trackId);
          var fExpr = (f.expression || "neutral").toUpperCase();
          var fConf = Math.round((f.confidence || 0) * 100);
          var fStatus = f.status || "face_detected";

          html +=
            '<div class="face-person-card" style="border-left-color: ' + theme.border + ';">' +
              '<div class="face-person-card-header">' +
                '<span class="face-person-title">' +
                  '<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ' + theme.border + ';"></span>' +
                  'Person #' + trackId +
                '</span>' +
                '<span class="face-person-badge" style="background: ' + theme.fill + '; color: ' + theme.stroke + ';">ID #' + trackId + ' · ' + escapeHtml(fStatus) + '</span>' +
              '</div>' +
              '<div class="face-person-body">' +
                '<span class="face-person-expression" style="color: ' + theme.stroke + ';">' + escapeHtml(fExpr) + '</span>' +
                '<strong style="font-size: 12px; color: var(--text);">' + fConf + '%</strong>' +
              '</div>' +
              '<div class="face-person-bar">' +
                '<div class="face-person-fill" style="width: ' + fConf + '%; background: ' + theme.stroke + ';"></div>' +
              '</div>' +
            '</div>';
        });
        multiContainer.innerHTML = html;
      }
    }

    // 5. Active category highlight chips
    var activeSet = {};
    faces.forEach(function (f) {
      if (f.expression) activeSet[f.expression.toLowerCase()] = true;
    });
    if (res.expression) activeSet[res.expression.toLowerCase()] = true;

    chips.forEach(function (chip) {
      var cat = chip.dataset.category;
      chip.classList.toggle("active", Boolean(activeSet[cat]));
    });

    // 6. Draw overlay bounding boxes
    drawFaceOverlays(faces, 640, 480);
  }

  function runFaceStep() {
    if (!faceMedia.stream || !faceMedia.video || faceMedia.isProcessing) return;
    var base64 = captureFrameBase64(faceMedia.video, faceMedia.canvas);
    if (!base64) return;

    faceMedia.isProcessing = true;
    SignBridgeAPI.analyzeFace(base64)
      .then(function (res) {
        updateFaceHUD(res);
      })
      .catch(function (err) {
        var status = $("#faceStatusDetail");
        if (status) status.textContent = "Facial analysis error: " + err.message;
      })
      .finally(function () {
        faceMedia.isProcessing = false;
      });
  }

  function startFaceCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("Webcam is not supported in this browser.", "error");
      return;
    }
    var startBtn = $("#faceCamStart");
    var stopBtn = $("#faceCamStop");
    var analyzeBtn = $("#faceAnalyzeBtn");

    if (startBtn) startBtn.disabled = true;
    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    })
      .then(function (stream) {
        faceMedia.stream = stream;
        faceMedia.video.srcObject = stream;
        $("#faceCamEmpty").hidden = true;
        if (stopBtn) stopBtn.disabled = false;
        if (analyzeBtn) analyzeBtn.disabled = false;
        toast("Facial camera active", "success");

        if (faceMedia.loopIntervalId) clearInterval(faceMedia.loopIntervalId);
        faceMedia.loopIntervalId = setInterval(function () {
          var toggle = $("#faceContinuousToggle");
          if (toggle && toggle.checked && faceMedia.stream) {
            runFaceStep();
          }
        }, 600);
      })
      .catch(function (err) {
        if (startBtn) startBtn.disabled = false;
        toast("Camera error: " + err.message, "error");
      });
  }

  function stopFaceCamera() {
    if (faceMedia.loopIntervalId) {
      clearInterval(faceMedia.loopIntervalId);
      faceMedia.loopIntervalId = null;
    }
    if (faceMedia.stream) {
      faceMedia.stream.getTracks().forEach(function (t) { t.stop(); });
      faceMedia.stream = null;
    }
    if (faceMedia.video) faceMedia.video.srcObject = null;
    if (faceMedia.overlayCanvas) {
      var ctx = faceMedia.overlayCanvas.getContext("2d");
      ctx.clearRect(0, 0, faceMedia.overlayCanvas.width, faceMedia.overlayCanvas.height);
    }
    var empty = $("#faceCamEmpty");
    if (empty) empty.hidden = false;
    var startBtn = $("#faceCamStart");
    var stopBtn = $("#faceCamStop");
    var analyzeBtn = $("#faceAnalyzeBtn");

    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (analyzeBtn) analyzeBtn.disabled = true;
    faceMedia.isProcessing = false;
  }

  function initFace() {
    faceMedia.video = $("#faceCamVideo");
    faceMedia.canvas = $("#faceCaptureCanvas");
    faceMedia.overlayCanvas = $("#faceOverlayCanvas");

    var startBtn = $("#faceCamStart");
    var stopBtn = $("#faceCamStop");
    var analyzeBtn = $("#faceAnalyzeBtn");

    if (startBtn) startBtn.addEventListener("click", startFaceCamera);
    if (stopBtn) stopBtn.addEventListener("click", stopFaceCamera);
    if (analyzeBtn) analyzeBtn.addEventListener("click", runFaceStep);
  }

  /* ---------- 9. Settings Module ---------- */
  function fillSettingsForm() {
    var url = $("#backendUrl");
    var epH = $("#epHealth");
    var epP = $("#epPredict");
    var epF = $("#epFace");
    var tOut = $("#timeoutSec");

    if (url) url.value = settings.backendUrl;
    if (epH) epH.value = settings.endpoints.health;
    if (epP) epP.value = settings.endpoints.predict;
    if (epF) epF.value = settings.endpoints.face;
    if (tOut) tOut.value = settings.timeoutSec;
  }

  function initSettings() {
    fillSettingsForm();
    var form = $("#settingsForm");
    var testBtn = $("#testConnection");
    var resetBtn = $("#resetSettings");

    if (form) {
      form.addEventListener("submit", function (e) {
        e.preventDefault();
        var url = ($("#backendUrl").value || "").trim();
        if (url && !/^https?:\/\//i.test(url)) url = "http://" + url;
        settings.backendUrl = url;
        settings.endpoints.health = ($("#epHealth").value || "").trim() || DEFAULTS.endpoints.health;
        settings.endpoints.predict = ($("#epPredict").value || "").trim() || DEFAULTS.endpoints.predict;
        settings.endpoints.face = ($("#epFace").value || "").trim() || DEFAULTS.endpoints.face;
        settings.timeoutSec = parseInt($("#timeoutSec").value, 10) || DEFAULTS.timeoutSec;

        saveSettings(settings);
        toast("Settings saved", "success");
        checkBackendConnection(true);
      });
    }

    if (testBtn) {
      testBtn.addEventListener("click", function () {
        checkBackendConnection(true);
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener("click", function () {
        settings = Object.assign({}, DEFAULTS);
        saveSettings(settings);
        fillSettingsForm();
        toast("Settings reset to defaults");
        checkBackendConnection(true);
      });
    }
  }

  /* ---------- 10. Diagnostics ---------- */
  function runSystemChecks() {
    var camCheck = $('[data-check="cam"]');
    var speechCheck = $('[data-check="speech"]');
    var backendCheck = $('[data-check="backend"]');
    var mediapipeCheck = $('[data-check="mediapipe"]');
    var facialCheck = $('[data-check="facial"]');

    function setCheck(li, ok, val) {
      if (!li) return;
      li.className = ok ? "ok" : "fail";
      var valEl = $(".check-val", li);
      if (valEl) valEl.textContent = val;
    }

    // Camera check
    var hasCam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    setCheck(camCheck, hasCam, hasCam ? "Available" : "Not supported");

    // Speech synthesis check
    var hasSpeech = !!("speechSynthesis" in window);
    setCheck(speechCheck, hasSpeech, hasSpeech ? "Available" : "Not supported");

    // Backend and AI check
    SignBridgeAPI.checkHealth()
      .then(function (res) {
        setCheck(backendCheck, true, "Online (" + (res.version || "2.0") + ")");
        setCheck(mediapipeCheck, true, res.detector_backend || "mediapipe_tasks");
        setCheck(facialCheck, true, res.facial_module_status || "Ready");
      })
      .catch(function () {
        setCheck(backendCheck, false, "Offline");
        setCheck(mediapipeCheck, false, "Awaiting backend");
        setCheck(facialCheck, false, "Awaiting backend");
      });
  }

  function startClock() {
    var el = $("#clock");
    if (!el) return;
    function tick() {
      var d = new Date();
      el.textContent = d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }) + " · " + pad(d.getHours()) + ":" + pad(d.getMinutes());
    }
    tick();
    setInterval(tick, 15000);
  }

  /* ---------- 11. App Initialization ---------- */
  document.addEventListener("DOMContentLoaded", function () {
    initNavigation();
    initSign();
    initTTS();
    initFace();
    initHistory();
    initSettings();
    startClock();
    updateStatsDisplay();

    var rerunBtn = $("#rerunChecks");
    if (rerunBtn) rerunBtn.addEventListener("click", runSystemChecks);

    var clearActBtn = $("#clearActivity");
    if (clearActBtn) {
      clearActBtn.addEventListener("click", function () {
        var list = $("#activityList");
        if (list) list.innerHTML = '<li class="activity-empty">No activity logged yet.</li>';
      });
    }

    // Initial check
    checkBackendConnection(false).finally(runSystemChecks);
  });
})();

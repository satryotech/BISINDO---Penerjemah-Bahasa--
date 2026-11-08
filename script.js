import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);

const camera = $("camera");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const cameraEmpty = $("cameraEmpty");
const cameraStage = $("cameraStage");
const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const speakBtn = $("speakBtn");
const clearSpeechBtn = $("clearSpeechBtn");
const cameraPill = $("cameraPill");
const cameraOverlayText = $("cameraOverlayText");
const handCount = $("handCount");
const modelState = $("modelState");
const gestureResult = $("gestureResult");
const detectionHint = $("detectionHint");
const detectionIndicator = $("detectionIndicator");
const translationText = $("translationText");
const translationFootnote = $("translationFootnote");
const systemMessage = $("systemMessage");
const logList = $("logList");

let landmarker = null;
let drawingUtils = null;
let stream = null;
let animationId = null;
let running = false;
let lastVideoTime = -1;
let busy = false;

function log(message, level = "info") {
  const entry = document.createElement("div");
  entry.className = "log-entry";

  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = new Date().toLocaleTimeString("id-ID", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });

  const content = document.createElement("span");
  content.textContent = message;
  if (level === "error") content.style.color = "#ffaaa5";

  entry.append(time, content);
  logList.prepend(entry);

  while (logList.children.length > 30) {
    logList.lastElementChild.remove();
  }
}

function setMessage(message) {
  systemMessage.textContent = message;
}

async function loadModel() {
  if (landmarker) return;

  modelState.textContent = "Memuat...";
  log("Memuat library dan model pelacakan tangan.");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  try {
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
    });
  } catch (gpuError) {
    // Fallback CPU jika GPU tidak tersedia atau gagal diinisialisasi.
    log("GPU tidak tersedia; mencoba mode CPU.", "info");
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
    });
  }

  drawingUtils = new DrawingUtils(ctx);
  modelState.textContent = "Siap";
  log("Model deteksi tangan berhasil dimuat.");
}

async function startCamera() {
  if (running || busy) return;
  busy = true;
  startBtn.disabled = true;
  setMessage("Menyiapkan model dan meminta izin kamera...");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Kamera tidak tersedia. Buka melalui localhost atau HTTPS.");
    }

    await loadModel();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });

    camera.srcObject = stream;
    await new Promise((resolve) => {
      if (camera.readyState >= 2) return resolve();
      camera.onloadedmetadata = () => resolve();
    });
    await camera.play();

    canvas.width = camera.videoWidth || 640;
    canvas.height = camera.videoHeight || 480;
    cameraEmpty.hidden = true;
    cameraPill.classList.add("active");
    cameraPill.innerHTML = '<span class="status-dot"></span> Kamera aktif';
    cameraOverlayText.textContent = "PELACAKAN AKTIF";
    running = true;
    stopBtn.disabled = false;
    handCount.textContent = "0";
    setMessage("Kamera aktif. Tunjukkan tangan di depan kamera.");
    log("Kamera aktif. Video diproses pada perangkat ini.");
    renderLoop();
  } catch (error) {
    console.error(error);
    const message = explainError(error);
    setMessage(message);
    log(message, "error");
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    camera.srcObject = null;
    startBtn.disabled = false;
  } finally {
    busy = false;
  }
}

function explainError(error) {
  if (error.name === "NotAllowedError") {
    return "Izin kamera ditolak. Izinkan kamera melalui pengaturan browser.";
  }
  if (error.name === "NotFoundError") {
    return "Kamera tidak ditemukan. Sambungkan atau pilih kamera lain.";
  }
  if (error.name === "NotReadableError") {
    return "Kamera sedang dipakai aplikasi lain. Tutup aplikasi tersebut lalu coba lagi.";
  }
  return `Gagal memulai: ${error.message || "kesalahan tidak diketahui"}. Periksa internet dan konsol browser.`;
}

function renderLoop() {
  if (!running) return;

  if (camera.readyState >= 2 && camera.currentTime !== lastVideoTime) {
    lastVideoTime = camera.currentTime;
    try {
      const result = landmarker.detectForVideo(camera, performance.now());
      drawResult(result);
      updateStatus(result);
    } catch (error) {
      console.error("Deteksi gagal:", error);
      setMessage("Deteksi mengalami kendala. Coba muat ulang halaman.");
      log("Terjadi kesalahan saat memproses frame.", "error");
      stopCamera();
      return;
    }
  }

  animationId = requestAnimationFrame(renderLoop);
}

function drawResult(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!result.landmarks?.length) return;

  // Karena video dicerminkan dengan CSS, koordinat overlay juga dicerminkan.
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  for (const landmarks of result.landmarks) {
    drawingUtils.drawConnectors(
      landmarks,
      HandLandmarker.HAND_CONNECTIONS,
      { color: "#6ce7d0", lineWidth: 3 }
    );
    drawingUtils.drawLandmarks(landmarks, {
      color: "#f5fbff",
      fillColor: "#438cf3",
      lineWidth: 1,
      radius: 3
    });
  }
  ctx.restore();
}

function updateStatus(result) {
  const count = result.landmarks?.length || 0;
  handCount.textContent = String(count);

  if (count === 0) {
    gestureResult.textContent = "Tangan belum terdeteksi";
    detectionHint.textContent = "Coba dekatkan tangan dan pastikan pencahayaan cukup.";
    detectionIndicator.classList.remove("detected");
    translationText.textContent = "Belum ada hasil terjemahan.";
    translationFootnote.textContent = "Menunggu tangan terdeteksi";
    speakBtn.disabled = true;
    return;
  }

  gestureResult.textContent = count === 1
    ? "1 tangan terdeteksi"
    : `${count} tangan terdeteksi`;
  detectionHint.textContent = "Titik tangan berhasil dilacak. Makna isyarat belum dikenali.";
  detectionIndicator.classList.add("detected");
  translationText.textContent =
    "Pelacakan berhasil — model penerjemah BISINDO belum terpasang.";
  translationFootnote.textContent =
    "Landmark ≠ terjemahan BISINDO";
  speakBtn.disabled = false;
}

function speakText(text) {
  if (!("speechSynthesis" in window)) {
    setMessage("Browser ini tidak mendukung fitur text-to-speech.");
    log("Text-to-speech tidak didukung browser.", "error");
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "id-ID";
  utterance.rate = 0.92;
  utterance.pitch = 1;
  utterance.onstart = () => setMessage("Suara sedang diputar.");
  utterance.onend = () => setMessage("Suara selesai diputar.");
  utterance.onerror = () => setMessage("Suara tidak dapat diputar pada perangkat ini.");
  window.speechSynthesis.speak(utterance);
  log("Memutar suara pemberitahuan prototype.");
}

function stopCamera() {
  running = false;
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  camera.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  cameraEmpty.hidden = false;
  cameraPill.classList.remove("active");
  cameraPill.innerHTML = '<span class="status-dot"></span> Kamera mati';
  cameraOverlayText.textContent = "MENUNGGU KAMERA";
  handCount.textContent = "—";
  gestureResult.textContent = "Menunggu kamera";
  detectionHint.textContent = "Aktifkan kamera untuk mulai melacak titik-titik tangan.";
  detectionIndicator.classList.remove("detected");
  translationText.textContent = "Hasil terjemahan akan muncul di sini.";
  translationFootnote.textContent = "Model BISINDO belum terpasang";
  speakBtn.disabled = true;
  stopBtn.disabled = true;
  startBtn.disabled = false;
  window.speechSynthesis?.cancel();
  setMessage("Kamera dimatikan.");
  log("Kamera dimatikan oleh pengguna.");
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

speakBtn.addEventListener("click", () => {
  speakText(
    "Ini adalah pemberitahuan prototype. Tangan berhasil terdeteksi, tetapi terjemahan BISINDO belum tersedia."
  );
});

clearSpeechBtn.addEventListener("click", () => {
  window.speechSynthesis?.cancel();
  setMessage("Pemutaran suara dihentikan.");
  log("Pemutaran suara dihentikan.");
});

$("clearLogBtn").addEventListener("click", () => {
  logList.replaceChildren();
  log("Log aktivitas dibersihkan.");
});

const helpDialog = $("helpDialog");
$("helpBtn").addEventListener("click", () => helpDialog.showModal());

window.addEventListener("pagehide", () => {
  running = false;
  if (animationId !== null) cancelAnimationFrame(animationId);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  window.speechSynthesis?.cancel();
});

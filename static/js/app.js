const statusPill = document.getElementById('statusPill');
const overlay = document.getElementById('overlay');
const overlayTitle = document.getElementById('overlayTitle');
const overlayMessage = document.getElementById('overlayMessage');
const refreshButton = document.getElementById('refreshButton');
const videoStream = document.getElementById('videoStream');

const HEALTH_CHECK_INTERVAL = 5000;
let isHealthy = false;
let healthTimer = null;

function setStatus(text, type) {
  statusPill.textContent = text;
  statusPill.classList.remove('status-loading', 'status-ok', 'status-error');
  statusPill.classList.add(`status-${type}`);
}

function showOverlay(title, message) {
  overlay.classList.remove('hidden');
  overlayTitle.textContent = title;
  overlayMessage.textContent = message;
}

function hideOverlay() {
  overlay.classList.add('hidden');
}

function reloadStream() {
  videoStream.src = '/video_feed?reload=' + Date.now();
  showOverlay('Reloading stream...', 'Please wait while the camera reconnects.');
  setStatus('Reconnecting...', 'loading');
}

async function checkHealth() {
  try {
    const response = await fetch('/health', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Health check failed');
    }

    if (!isHealthy) {
      isHealthy = true;
      setStatus('Live', 'ok');
      hideOverlay();
    }
  } catch (error) {
    isHealthy = false;
    setStatus('Offline', 'error');
    showOverlay('Camera unavailable', 'The app could not access the webcam. Try reconnecting your camera or pressing Retry.');
  }
}

videoStream.addEventListener('load', () => {
  setStatus('Live', 'ok');
  hideOverlay();
});

videoStream.addEventListener('error', () => {
  isHealthy = false;
  setStatus('Offline', 'error');
  showOverlay('Unable to load stream', 'The browser failed to receive the webcam feed. Use Retry or refresh the page.');
});

refreshButton.addEventListener('click', () => {
  reloadStream();
});

window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  healthTimer = setInterval(checkHealth, HEALTH_CHECK_INTERVAL);
});

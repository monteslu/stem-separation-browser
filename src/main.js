/**
 * Stem Separation POC
 * 
 * Browser-based audio stem separation using Demucs ONNX model
 */

// Unregister any stale service workers from previous dev sessions
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(registrations => {
    for (const registration of registrations) {
      registration.unregister();
      console.warn('Unregistered stale service worker:', registration.scope);
    }
  });
}

import { loadDemucsModel, separateStems, extractStems } from './demucs-loader.js';

// Global model cache
let demucsSession = null;
let modelLoading = false;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const statusContainer = document.getElementById('statusContainer');
const originalAudio = document.getElementById('originalAudio');
const originalPlayer = document.getElementById('originalPlayer');
const stemOutputs = document.getElementById('stemOutputs');

// State
let audioContext = null;

// Status helpers
function showStatus(message, type = 'loading') {
  statusContainer.innerHTML = `
    <div class="status ${type}">
      ${message}
      ${type === 'loading' ? '<div class="progress-bar"><div class="progress-fill" id="progressFill" style="width: 0%"></div></div>' : ''}
    </div>
  `;
}

function updateProgress(percent) {
  const fill = document.getElementById('progressFill');
  if (fill) fill.style.width = `${percent}%`;
}

function clearStatus() {
  statusContainer.innerHTML = '';
}

// Audio helpers
function getAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioContext;
}

async function loadAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const ctx = getAudioContext();
  return await ctx.decodeAudioData(arrayBuffer);
}

function audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const length = buffer.length * numChannels;
  const data = new Float32Array(length);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < buffer.length; i++) {
      data[i * numChannels + channel] = channelData[i];
    }
  }

  const dataLength = length * bytesPerSample;
  const wavBuffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(wavBuffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([wavBuffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function createAudioFromBuffer(audioBuffer) {
  const blob = audioBufferToWav(audioBuffer);
  return URL.createObjectURL(blob);
}

// Ensure model is loaded (auto-download, no prompts)
async function ensureModel() {
  if (demucsSession) return demucsSession;
  if (modelLoading) {
    // Wait for in-progress load
    while (modelLoading) {
      await new Promise(r => setTimeout(r, 500));
    }
    return demucsSession;
  }

  modelLoading = true;
  try {
    demucsSession = await loadDemucsModel((progress) => {
      if (progress.stage === 'downloading') {
        const mb = (progress.received / 1024 / 1024).toFixed(1);
        const totalMb = progress.total ? (progress.total / 1024 / 1024).toFixed(1) : '?';
        showStatus(`📥 Downloading Demucs model: ${mb}MB / ${totalMb}MB (${progress.percent}%)`, 'loading');
        updateProgress(progress.percent);
      } else if (progress.stage === 'loading') {
        showStatus('🔧 Initializing ONNX Runtime...', 'loading');
      }
    });
    return demucsSession;
  } finally {
    modelLoading = false;
  }
}

// Main processing function
async function processAudio(file) {
  try {
    showStatus('📂 Loading audio file...', 'loading');

    const audioBuffer = await loadAudioFile(file);

    // Show original
    originalAudio.style.display = 'block';
    originalPlayer.src = URL.createObjectURL(file);

    showStatus(`✅ Loaded: ${file.name} (${audioBuffer.duration.toFixed(1)}s, ${audioBuffer.sampleRate}Hz). Preparing model...`, 'loading');

    // Auto-download model if needed
    const session = await ensureModel();
    if (!session) {
      throw new Error('Failed to load Demucs model');
    }

    const duration = audioBuffer.duration.toFixed(0);
    showStatus(`🎵 Running stem separation on ${duration}s of audio... (this may take a while)`, 'loading');
    updateProgress(10);

    const result = await separateStems(session, audioBuffer, audioBuffer.sampleRate);
    
    showStatus('🎵 Extracting stems...', 'loading');
    updateProgress(90);
    const stems = extractStems(result, audioBuffer.sampleRate, getAudioContext());

    updateProgress(100);

    stemOutputs.innerHTML = `
      <div class="stem-card">
        <h3>🥁 Drums</h3>
        <audio controls src="${createAudioFromBuffer(stems.drums)}"></audio>
      </div>
      <div class="stem-card">
        <h3>🎸 Bass</h3>
        <audio controls src="${createAudioFromBuffer(stems.bass)}"></audio>
      </div>
      <div class="stem-card">
        <h3>🎹 Other</h3>
        <audio controls src="${createAudioFromBuffer(stems.other)}"></audio>
      </div>
      <div class="stem-card">
        <h3>🎤 Vocals</h3>
        <audio controls src="${createAudioFromBuffer(stems.vocals)}"></audio>
      </div>
    `;

    showStatus('✅ Stem separation complete!', 'success');

  } catch (error) {
    console.error('Processing error:', error);
    showStatus(`❌ Error: ${error.message}`, 'error');
  }
}

// Event handlers
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('audio/')) {
    processAudio(file);
  }
});
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) processAudio(file);
});

// Log info on load
console.warn('Stem Separation loaded');
console.warn('crossOriginIsolated:', typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : 'N/A');

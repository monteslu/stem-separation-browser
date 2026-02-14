/**
 * Stem Separation POC
 * 
 * Exploring browser-based audio stem separation using Transformers.js
 * 
 * Research notes:
 * - Transformers.js uses ONNX Runtime for inference
 * - Most source separation models (like Demucs) are PyTorch-based
 * - Need to find/convert models to ONNX format for browser use
 * - Alternative: Use simpler frequency-based separation as fallback
 */

// Unregister any stale service workers from previous dev sessions
// Service workers can cache HTML responses for .wasm URLs, breaking ONNX Runtime
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(registrations => {
    for (const registration of registrations) {
      registration.unregister();
      console.warn('Unregistered stale service worker:', registration.scope);
    }
  });
}

import { pipeline, env } from '@huggingface/transformers';
import { loadDemucsModel, separateStems, extractStems } from './demucs-loader.js';

// Configure Transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

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
let currentAudioBuffer = null;

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
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  
  // Interleave channels
  const length = buffer.length * numChannels;
  const data = new Float32Array(length);
  
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < buffer.length; i++) {
      data[i * numChannels + channel] = channelData[i];
    }
  }
  
  // Create WAV file
  const dataLength = length * bytesPerSample;
  const wavBuffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(wavBuffer);
  
  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, 'WAVE');
  
  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  
  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);
  
  // Write audio data
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

// Simple frequency-based separation (fallback/demo)
function simpleFrequencySeparation(audioBuffer) {
  const ctx = getAudioContext();
  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length;
  const channels = audioBuffer.numberOfChannels;
  
  // Create output buffers
  const vocalBuffer = ctx.createBuffer(channels, length, sampleRate);
  const instrumentBuffer = ctx.createBuffer(channels, length, sampleRate);
  
  // Simple approach: vocals tend to be center-panned
  // Subtract left from right to get some vocal separation
  // This is a VERY crude approximation
  
  if (channels >= 2) {
    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.getChannelData(1);
    
    const vocalLeft = vocalBuffer.getChannelData(0);
    const vocalRight = vocalBuffer.getChannelData(1);
    const instLeft = instrumentBuffer.getChannelData(0);
    const instRight = instrumentBuffer.getChannelData(1);
    
    for (let i = 0; i < length; i++) {
      // Center extraction (crude vocal isolation)
      const mid = (left[i] + right[i]) / 2;
      const side = (left[i] - right[i]) / 2;
      
      // Vocals: mostly mid (center) content
      vocalLeft[i] = mid * 0.8;
      vocalRight[i] = mid * 0.8;
      
      // Instruments: mostly side + some mid
      instLeft[i] = side + mid * 0.3;
      instRight[i] = -side + mid * 0.3;
    }
  } else {
    // Mono: can't do much without frequency separation
    const mono = audioBuffer.getChannelData(0);
    vocalBuffer.getChannelData(0).set(mono);
    instrumentBuffer.getChannelData(0).set(mono);
  }
  
  return { vocals: vocalBuffer, instruments: instrumentBuffer };
}

// ML-based separation attempt using Demucs ONNX
async function tryMLSeparation(audioBuffer) {
  // Check if model needs loading
  if (!demucsSession && !modelLoading) {
    showStatus(`
      <strong>🧠 Demucs ONNX Model Available!</strong><br><br>
      A pre-converted Demucs model (~180MB) was found on HuggingFace.<br>
      Click the button below to download and enable ML-powered separation.<br><br>
      <button id="loadModelBtn">📥 Load Demucs Model (~180MB)</button>
      <button id="skipModelBtn">⏭️ Skip (use demo mode)</button>
    `, 'info');
    
    return new Promise((resolve) => {
      document.getElementById('loadModelBtn')?.addEventListener('click', async () => {
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
          modelLoading = false;
          resolve(demucsSession);
        } catch (error) {
          modelLoading = false;
          console.error('Model load error:', error);
          showStatus(`❌ Failed to load model: ${error.message}`, 'error');
          resolve(null);
        }
      });
      
      document.getElementById('skipModelBtn')?.addEventListener('click', () => {
        resolve(null);
      });
    });
  }
  
  if (demucsSession) {
    // Model is loaded, run inference
    showStatus('🎵 Running Demucs stem separation...', 'loading');
    updateProgress(50);
    
    try {
      const output = await separateStems(demucsSession, audioBuffer, audioBuffer.sampleRate);
      const stems = extractStems(output, audioBuffer.sampleRate, getAudioContext());
      
      updateProgress(100);
      return stems;
    } catch (error) {
      console.error('Inference error:', error);
      showStatus(`❌ Inference failed: ${error.message}`, 'error');
      return null;
    }
  }
  
  return null; // Use fallback
}

// Main processing function
async function processAudio(file) {
  try {
    showStatus('📂 Loading audio file...', 'loading');
    
    currentAudioBuffer = await loadAudioFile(file);
    
    // Show original
    originalAudio.style.display = 'block';
    originalPlayer.src = URL.createObjectURL(file);
    
    showStatus(`✅ Loaded: ${file.name} (${currentAudioBuffer.duration.toFixed(1)}s, ${currentAudioBuffer.sampleRate}Hz)`, 'success');
    
    // Try ML separation first
    const mlResult = await tryMLSeparation(currentAudioBuffer);
    
    if (mlResult && mlResult.vocals) {
      // ML-based separation succeeded!
      stemOutputs.innerHTML = `
        <div class="info" style="border-left: 4px solid #32cd32;">
          <strong>✅ ML Separation Complete!</strong><br>
          Using Demucs neural network for high-quality stem separation.
        </div>
        <div class="stem-card">
          <h3>🥁 Drums</h3>
          <audio controls src="${createAudioFromBuffer(mlResult.drums)}"></audio>
        </div>
        <div class="stem-card">
          <h3>🎸 Bass</h3>
          <audio controls src="${createAudioFromBuffer(mlResult.bass)}"></audio>
        </div>
        <div class="stem-card">
          <h3>🎹 Other</h3>
          <audio controls src="${createAudioFromBuffer(mlResult.other)}"></audio>
        </div>
        <div class="stem-card">
          <h3>🎤 Vocals</h3>
          <audio controls src="${createAudioFromBuffer(mlResult.vocals)}"></audio>
        </div>
      `;
      
      showStatus('✅ ML stem separation complete!', 'success');
    } else {
      // Fall back to simple frequency separation
      showStatus('🎛️ Using frequency-based separation (demo mode)...', 'loading');
      updateProgress(50);
      
      await new Promise(r => setTimeout(r, 500)); // Small delay for UX
      
      const { vocals, instruments } = simpleFrequencySeparation(currentAudioBuffer);
      
      updateProgress(100);
      
      // Display results
      stemOutputs.innerHTML = `
        <div class="info">
          <strong>⚠️ Demo Mode:</strong> Using simple mid/side separation.<br>
          Load the Demucs model above for high-quality 4-stem separation!
        </div>
        <div class="stem-card">
          <h3>🎤 Vocals (center extraction)</h3>
          <audio controls src="${createAudioFromBuffer(vocals)}"></audio>
        </div>
        <div class="stem-card">
          <h3>🎸 Instruments (side extraction)</h3>
          <audio controls src="${createAudioFromBuffer(instruments)}"></audio>
        </div>
      `;
      
      showStatus('✅ Processing complete (demo mode)', 'success');
    }
    
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
console.warn('Stem Separation POC loaded');
console.warn('Using Transformers.js for potential ML-based separation');
console.warn('Fallback: frequency-based mid/side separation');
if (typeof crossOriginIsolated !== 'undefined') {
  console.warn('crossOriginIsolated:', crossOriginIsolated);
}

/**
 * Demucs ONNX Model Loader
 * 
 * Attempts to load and run pre-converted Demucs ONNX models from HuggingFace
 */

import * as ort from 'onnxruntime-web';

// Model URLs from HuggingFace
const MODELS = {
  htdemucs: 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx',
};

// Configure ONNX Runtime for browser environment
// Let ONNX Runtime auto-detect thread support based on crossOriginIsolated
if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
} else {
  // Without COOP/COEP headers, SharedArrayBuffer is unavailable
  ort.env.wasm.numThreads = 1;
}
ort.env.wasm.simd = true;

// Disable WebGPU — WASM backend is more reliable for this use case
async function getPreferredBackend() {
  console.warn('Using WASM backend for ONNX Runtime');
  return 'wasm';
}

// Load model
export async function loadDemucsModel(progressCallback) {
  const backend = await getPreferredBackend();
  
  progressCallback?.({ stage: 'downloading', percent: 0 });
  
  // Fetch with progress
  const response = await fetch(MODELS.htdemucs, { mode: 'cors' });
  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
  }
  const contentLength = response.headers.get('content-length');
  const total = parseInt(contentLength, 10) || 0;
  
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    progressCallback?.({ 
      stage: 'downloading', 
      percent: total ? Math.round((received / total) * 100) : 0,
      received,
      total
    });
  }
  
  // Combine chunks
  const modelBuffer = new Uint8Array(received);
  let position = 0;
  for (const chunk of chunks) {
    modelBuffer.set(chunk, position);
    position += chunk.length;
  }
  
  progressCallback?.({ stage: 'loading', percent: 0 });
  
  // Create session from ArrayBuffer (avoids MIME type issues with streaming)
  const session = await ort.InferenceSession.create(modelBuffer.buffer, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  });
  
  progressCallback?.({ stage: 'ready', percent: 100 });
  
  console.warn('Demucs model loaded!');
  console.warn('Input names:', session.inputNames);
  console.warn('Output names:', session.outputNames);
  
  return session;
}

// Run inference
export async function separateStems(session, audioData, sampleRate) {
  // Demucs expects:
  // - Input: [batch, channels, samples] as Float32
  // - Sample rate: 44100 Hz typically
  // - Output: [batch, stems, channels, samples]
  //   where stems = [drums, bass, other, vocals]
  
  const channels = audioData.numberOfChannels;
  const length = audioData.length;
  
  // Prepare input tensor
  // Interleave channels: [1, channels, samples]
  const inputData = new Float32Array(channels * length);
  
  for (let c = 0; c < channels; c++) {
    const channelData = audioData.getChannelData(c);
    for (let i = 0; i < length; i++) {
      inputData[c * length + i] = channelData[i];
    }
  }
  
  const inputTensor = new ort.Tensor('float32', inputData, [1, channels, length]);
  
  console.warn('Running inference...');
  console.warn('Input shape:', inputTensor.dims);
  
  // Run model
  const results = await session.run({ input: inputTensor });
  
  // Parse outputs
  const output = Object.values(results)[0];
  console.warn('Output shape:', output.dims);
  
  return output;
}

// Export stems as audio buffers
export function extractStems(outputTensor, sampleRate, audioContext) {
  const [batch, stems, channels, samples] = outputTensor.dims;
  const data = outputTensor.data;
  
  const stemNames = ['drums', 'bass', 'other', 'vocals'];
  const buffers = {};
  
  for (let s = 0; s < stems; s++) {
    const buffer = audioContext.createBuffer(channels, samples, sampleRate);
    
    for (let c = 0; c < channels; c++) {
      const channelData = buffer.getChannelData(c);
      const offset = s * channels * samples + c * samples;
      
      for (let i = 0; i < samples; i++) {
        channelData[i] = data[offset + i];
      }
    }
    
    buffers[stemNames[s]] = buffer;
  }
  
  return buffers;
}

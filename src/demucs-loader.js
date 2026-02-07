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

// Configure ONNX Runtime
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;

// Try WebGPU if available
async function getPreferredBackend() {
  if ('gpu' in navigator) {
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      if (adapter) {
        console.log('WebGPU available - using for acceleration');
        return 'webgpu';
      }
    } catch (e) {
      console.log('WebGPU not available:', e.message);
    }
  }
  console.log('Using WASM backend');
  return 'wasm';
}

// Load model
export async function loadDemucsModel(progressCallback) {
  const backend = await getPreferredBackend();
  
  progressCallback?.({ stage: 'downloading', percent: 0 });
  
  // Fetch with progress
  const response = await fetch(MODELS.htdemucs);
  const contentLength = response.headers.get('content-length');
  const total = parseInt(contentLength, 10);
  
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
      percent: Math.round((received / total) * 100),
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
  
  // Create session
  const session = await ort.InferenceSession.create(modelBuffer.buffer, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  });
  
  progressCallback?.({ stage: 'ready', percent: 100 });
  
  console.log('Demucs model loaded!');
  console.log('Input names:', session.inputNames);
  console.log('Output names:', session.outputNames);
  
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
  
  console.log('Running inference...');
  console.log('Input shape:', inputTensor.dims);
  
  // Run model
  const results = await session.run({ input: inputTensor });
  
  // Parse outputs
  const output = Object.values(results)[0];
  console.log('Output shape:', output.dims);
  
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

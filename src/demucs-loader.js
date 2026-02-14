/**
 * Demucs ONNX Model Loader
 * 
 * Loads and runs pre-converted Demucs ONNX models from HuggingFace.
 * The model expects two inputs:
 *   - "input": raw audio waveform [1, channels, samples]
 *   - "x": STFT spectrogram [1, channels, freq_bins, time_frames, 2] (real+imag)
 * 
 * STFT is computed in JS because torch STFT/iSTFT can't be exported to ONNX.
 * Reference: https://github.com/gianlourbano/demucs-onnx
 */

import * as ort from 'onnxruntime-web';

// Model URLs from HuggingFace
const MODELS = {
  htdemucs: 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx',
};

// STFT parameters matching Demucs defaults
const N_FFT = 4096;
const HOP_LENGTH = 1024;
const WIN_LENGTH = 4096;

// Configure ONNX Runtime
if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
  ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 2);
} else {
  ort.env.wasm.numThreads = 1;
}
ort.env.wasm.simd = true;

/**
 * Compute STFT of audio data (real + imaginary components)
 * Returns Float32Array with shape [channels, freq_bins, time_frames, 2]
 */
function computeSTFT(audioData, channels, length) {
  const freqBins = Math.floor(N_FFT / 2) + 1; // 2049
  const timeFrames = Math.floor((length - N_FFT) / HOP_LENGTH) + 1;
  
  // Hann window
  const window = new Float32Array(WIN_LENGTH);
  for (let i = 0; i < WIN_LENGTH; i++) {
    window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / WIN_LENGTH));
  }
  
  const stftData = new Float32Array(channels * freqBins * timeFrames * 2);
  
  for (let c = 0; c < channels; c++) {
    const channelData = audioData.getChannelData(c);
    
    for (let t = 0; t < timeFrames; t++) {
      const start = t * HOP_LENGTH;
      
      // Apply window and compute FFT for this frame
      const real = new Float32Array(N_FFT);
      const imag = new Float32Array(N_FFT);
      
      for (let i = 0; i < WIN_LENGTH; i++) {
        if (start + i < length) {
          real[i] = channelData[start + i] * window[i];
        }
      }
      
      // In-place FFT (Cooley-Tukey radix-2)
      fft(real, imag, N_FFT);
      
      // Store only positive frequencies (0 to N_FFT/2 inclusive)
      for (let f = 0; f < freqBins; f++) {
        const idx = ((c * freqBins + f) * timeFrames + t) * 2;
        stftData[idx] = real[f];     // real part
        stftData[idx + 1] = imag[f]; // imaginary part
      }
    }
  }
  
  return { data: stftData, freqBins, timeFrames };
}

/**
 * In-place Cooley-Tukey FFT (radix-2, DIT)
 */
function fft(real, imag, n) {
  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }
  
  // Butterfly operations
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = -2 * Math.PI / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);
    
    for (let i = 0; i < n; i += len) {
      let curReal = 1, curImag = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k;
        const oddIdx = i + k + halfLen;
        
        const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
        const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];
        
        real[oddIdx] = real[evenIdx] - tReal;
        imag[oddIdx] = imag[evenIdx] - tImag;
        real[evenIdx] += tReal;
        imag[evenIdx] += tImag;
        
        const newCurReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = newCurReal;
      }
    }
  }
}

// Load model
export async function loadDemucsModel(progressCallback) {
  progressCallback?.({ stage: 'downloading', percent: 0 });
  
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
  
  const modelBuffer = new Uint8Array(received);
  let position = 0;
  for (const chunk of chunks) {
    modelBuffer.set(chunk, position);
    position += chunk.length;
  }
  
  progressCallback?.({ stage: 'loading', percent: 0 });
  
  const session = await ort.InferenceSession.create(modelBuffer.buffer, {
    executionProviders: ['wasm'],
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
  const channels = audioData.numberOfChannels;
  const length = audioData.length;
  
  // Pad/truncate to a length that's a power-of-2 friendly for STFT
  // htdemucs_embedded model expects exactly 343980 samples (~7.8s at 44.1kHz)
  const CHUNK_SIZE = 343980;
  const processLength = Math.min(length, CHUNK_SIZE);
  
  // Prepare raw audio tensor: [1, channels, samples]
  const inputData = new Float32Array(channels * processLength);
  for (let c = 0; c < channels; c++) {
    const channelData = audioData.getChannelData(c);
    for (let i = 0; i < processLength; i++) {
      inputData[c * processLength + i] = channelData[i];
    }
  }
  
  const inputTensor = new ort.Tensor('float32', inputData, [1, channels, processLength]);
  
  // Compute STFT for second input
  console.warn('Computing STFT...');
  const { data: stftData, freqBins, timeFrames } = computeSTFT(audioData, channels, processLength);
  const stftTensor = new ort.Tensor('float32', stftData, [1, channels, freqBins, timeFrames, 2]);
  
  console.warn('Running inference...');
  console.warn('Input "input" shape:', inputTensor.dims);
  console.warn('Input "x" shape:', stftTensor.dims);
  
  // Build feeds using actual input names from the model
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  feeds[session.inputNames[1]] = stftTensor;
  
  const results = await session.run(feeds);
  
  const output = results[session.outputNames[0]];
  console.warn('Output shape:', output.dims);
  
  return { output, processLength };
}

// Export stems as audio buffers
export function extractStems(result, sampleRate, audioContext) {
  const { output, processLength } = result;
  const dims = output.dims;
  const data = output.data;
  
  // Output shape should be [1, stems, channels, samples] or [stems, channels, samples]
  let stems, channels, samples;
  if (dims.length === 4) {
    [, stems, channels, samples] = dims;
  } else if (dims.length === 3) {
    [stems, channels, samples] = dims;
  } else {
    console.warn('Unexpected output shape:', dims);
    throw new Error(`Unexpected output tensor shape: [${dims.join(', ')}]`);
  }
  
  const stemNames = ['drums', 'bass', 'other', 'vocals'];
  const buffers = {};
  
  for (let s = 0; s < Math.min(stems, 4); s++) {
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

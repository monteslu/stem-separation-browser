/**
 * Demucs ONNX Model Loader
 * 
 * Loads and runs pre-converted Demucs ONNX models from HuggingFace.
 * The model expects two inputs:
 *   - "input": raw audio waveform [1, channels, samples]
 *   - "x": STFT spectrogram [1, channels*2, n_fft//2, time_frames]
 *     where channels*2 = [ch0_real, ch0_imag, ch1_real, ch1_imag]
 * 
 * STFT must match PyTorch: center=True, normalized=True, hann_window,
 * pad_mode='reflect', n_fft=4096, hop_length=1024
 * 
 * Reference: https://github.com/gianlourbano/demucs-onnx
 * PyTorch STFT: https://github.com/facebookresearch/demucs (demucs/spec.py)
 */

import * as ort from 'onnxruntime-web';

// Model URLs from HuggingFace
const MODELS = {
  htdemucs: 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx',
};

// STFT parameters matching Demucs defaults (htdemucs.py + spec.py)
const N_FFT = 4096;
const HOP_LENGTH = 1024; // n_fft // 4
const WIN_LENGTH = 4096;
const FREQ_BINS = 2048;  // n_fft // 2 (drop Nyquist bin)
const NORM_FACTOR = Math.sqrt(N_FFT); // PyTorch normalized=True

// Configure ONNX Runtime
if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
  ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 2);
} else {
  ort.env.wasm.numThreads = 1;
}
ort.env.wasm.simd = true;

/**
 * Compute STFT matching PyTorch's torch.stft with:
 *   center=True, normalized=True, hann_window, pad_mode='reflect'
 * 
 * Returns rank-4 data: [channels*2, FREQ_BINS, time_frames]
 * where channels*2 interleaves real/imag per channel
 */
function computeSTFT(getChannelData, channels, length) {
  // center=True: frames = floor(length / hop) + 1
  const timeFrames = Math.floor(length / HOP_LENGTH) + 1;
  const pad = Math.floor(N_FFT / 2);
  
  // Hann window (matching PyTorch's torch.hann_window)
  const window = new Float32Array(WIN_LENGTH);
  for (let i = 0; i < WIN_LENGTH; i++) {
    window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / WIN_LENGTH));
  }
  
  const totalChannels = channels * 2; // real + imag per channel
  const stftData = new Float32Array(totalChannels * FREQ_BINS * timeFrames);
  
  for (let c = 0; c < channels; c++) {
    const channelData = getChannelData(c);
    
    for (let t = 0; t < timeFrames; t++) {
      // center=True: frame center at t * HOP_LENGTH
      const frameStart = t * HOP_LENGTH - pad;
      
      const real = new Float32Array(N_FFT);
      const imag = new Float32Array(N_FFT);
      
      for (let i = 0; i < WIN_LENGTH; i++) {
        let idx = frameStart + i;
        // Reflect padding (PyTorch pad_mode='reflect')
        if (idx < 0) idx = -idx;
        if (idx >= length) idx = 2 * length - idx - 2;
        // Clamp for safety
        idx = Math.max(0, Math.min(length - 1, idx));
        real[i] = channelData[idx] * window[i];
      }
      
      // In-place radix-2 FFT
      fft(real, imag, N_FFT);
      
      // Normalize (PyTorch normalized=True divides by sqrt(n_fft))
      const realChIdx = c * 2;
      const imagChIdx = c * 2 + 1;
      for (let f = 0; f < FREQ_BINS; f++) {
        // Bins 0..2047 (keep DC, drop Nyquist at bin 2048)
        // htdemucs uses freqs = nfft // 2 = 2048, slicing [:2048] from stft output
        stftData[(realChIdx * FREQ_BINS + f) * timeFrames + t] = real[f] / NORM_FACTOR;
        stftData[(imagChIdx * FREQ_BINS + f) * timeFrames + t] = imag[f] / NORM_FACTOR;
      }
    }
  }
  
  return { data: stftData, totalChannels, timeFrames };
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

// Process a single chunk through the model
async function processChunk(session, audioData, channels, offset, chunkSize) {
  const length = audioData.length;
  
  // Prepare raw audio tensor: [1, channels, chunkSize]
  const inputData = new Float32Array(channels * chunkSize);
  for (let c = 0; c < channels; c++) {
    const channelData = audioData.getChannelData(c);
    for (let i = 0; i < chunkSize; i++) {
      const srcIdx = offset + i;
      if (srcIdx >= 0 && srcIdx < length) {
        inputData[c * chunkSize + i] = channelData[srcIdx];
      }
    }
  }
  
  const inputTensor = new ort.Tensor('float32', inputData, [1, channels, chunkSize]);
  
  // Compute STFT matching PyTorch spec.py
  const getChannelData = (c) => inputData.subarray(c * chunkSize, (c + 1) * chunkSize);
  const { data: stftData, totalChannels, timeFrames } = computeSTFT(getChannelData, channels, chunkSize);
  const stftTensor = new ort.Tensor('float32', stftData, [1, totalChannels, FREQ_BINS, timeFrames]);
  
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  feeds[session.inputNames[1]] = stftTensor;
  
  const results = await session.run(feeds);
  
  // Use time-domain output [1, stems, channels, samples]
  return results[session.outputNames[1]];
}

// Run inference with chunked processing for full-length audio
export async function separateStems(session, audioData, sampleRate, onProgress) {
  const channels = audioData.numberOfChannels;
  const length = audioData.length;
  
  // htdemucs_embedded model expects exactly 343980 samples per chunk (~7.8s at 44.1kHz)
  const CHUNK_SIZE = 343980;
  const OVERLAP = Math.floor(CHUNK_SIZE * 0.25); // 25% overlap
  const STEP = CHUNK_SIZE - OVERLAP;
  const NUM_STEMS = 4;
  
  const numChunks = Math.max(1, Math.ceil((length - OVERLAP) / STEP));
  
  console.warn(`Processing ${numChunks} chunk(s) of ${CHUNK_SIZE} samples (${(CHUNK_SIZE/sampleRate).toFixed(1)}s each)`);
  console.warn(`Total audio: ${(length/sampleRate).toFixed(1)}s, overlap: ${(OVERLAP/sampleRate).toFixed(1)}s`);
  
  // Allocate output buffers
  const outputLength = Math.max(length, CHUNK_SIZE);
  const stemData = new Array(NUM_STEMS);
  const weightSum = new Float32Array(outputLength);
  for (let s = 0; s < NUM_STEMS; s++) {
    stemData[s] = new Array(channels);
    for (let c = 0; c < channels; c++) {
      stemData[s][c] = new Float32Array(outputLength);
    }
  }
  
  // Crossfade window
  const fadeLen = OVERLAP;
  const fadeWindow = new Float32Array(CHUNK_SIZE);
  for (let i = 0; i < CHUNK_SIZE; i++) {
    if (i < fadeLen) {
      fadeWindow[i] = i / fadeLen;
    } else if (i >= CHUNK_SIZE - fadeLen) {
      fadeWindow[i] = (CHUNK_SIZE - i) / fadeLen;
    } else {
      fadeWindow[i] = 1.0;
    }
  }
  
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const offset = chunk * STEP;
    const pct = Math.round(((chunk + 1) / numChunks) * 100);
    console.warn(`Chunk ${chunk + 1}/${numChunks} (offset: ${offset}, ${pct}%)`);
    onProgress?.({ chunk: chunk + 1, total: numChunks, percent: pct });
    
    const output = await processChunk(session, audioData, channels, offset, CHUNK_SIZE);
    const data = output.data;
    const [, stems, ch, samples] = output.dims;
    
    for (let s = 0; s < stems; s++) {
      for (let c = 0; c < ch; c++) {
        for (let i = 0; i < samples; i++) {
          const outIdx = offset + i;
          if (outIdx < outputLength) {
            let weight = fadeWindow[i];
            if (chunk === 0 && i < fadeLen) weight = 1.0;
            if (chunk === numChunks - 1 && i >= CHUNK_SIZE - fadeLen) weight = 1.0;
            
            const srcIdx = s * ch * samples + c * samples + i;
            stemData[s][c][outIdx] += data[srcIdx] * weight;
            if (s === 0 && c === 0) weightSum[outIdx] += weight;
          }
        }
      }
    }
  }
  
  // Normalize by weight sum
  for (let s = 0; s < NUM_STEMS; s++) {
    for (let c = 0; c < channels; c++) {
      for (let i = 0; i < length; i++) {
        if (weightSum[i] > 0) {
          stemData[s][c][i] /= weightSum[i];
        }
      }
    }
  }
  
  return { stemData, channels, length, numStems: NUM_STEMS };
}

// Export stems as audio buffers
export function extractStems(result, sampleRate, audioContext) {
  const { stemData, channels, length, numStems } = result;
  
  const stemNames = ['drums', 'bass', 'other', 'vocals'];
  const buffers = {};
  
  for (let s = 0; s < numStems; s++) {
    const buffer = audioContext.createBuffer(channels, length, sampleRate);
    
    for (let c = 0; c < channels; c++) {
      buffer.getChannelData(c).set(stemData[s][c].subarray(0, length));
    }
    
    buffers[stemNames[s]] = buffer;
  }
  
  return buffers;
}

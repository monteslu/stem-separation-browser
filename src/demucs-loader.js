/**
 * Demucs ONNX Model Loader
 * 
 * Loads and runs Demucs ONNX models from HuggingFace.
 * 
 * The STFT preprocessing must exactly match the standalone_spec() function
 * from sevagh/demucs.onnx (demucs-for-onnx/demucs/htdemucs.py):
 * 
 *   1. Compute le = ceil(length / hop_length)
 *   2. Pad signal with reflect: (pad, pad + le*hl - length) where pad = hl//2*3
 *   3. Run STFT with center=True, normalized=True, hann_window(nfft)
 *   4. Drop last freq bin: z[..., :-1, :]
 *   5. Trim time frames: z[..., 2:2+le]
 *   6. CaC (complex-as-channels): reshape [B,C,Fr,T] complex -> [B,C*2,Fr,T] real
 *      interleaved as [ch0_real, ch0_imag, ch1_real, ch1_imag]
 * 
 * Reference: https://github.com/sevagh/demucs.onnx
 */

import * as ort from 'onnxruntime-web';

const MODELS = {
  htdemucs: 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx',
};

const N_FFT = 4096;
const HOP_LENGTH = 1024; // nfft // 4
const NORM_FACTOR = Math.sqrt(N_FFT); // PyTorch normalized=True

// Configure ONNX Runtime
if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
  ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 2);
} else {
  ort.env.wasm.numThreads = 1;
}
ort.env.wasm.simd = true;

/**
 * Reflect-pad a Float32Array on both sides
 */
function reflectPad(data, padLeft, padRight) {
  const len = data.length;
  const out = new Float32Array(padLeft + len + padRight);
  // Copy original
  out.set(data, padLeft);
  // Reflect left
  for (let i = 0; i < padLeft; i++) {
    const srcIdx = padLeft - i; // reflect index
    out[i] = srcIdx < len ? data[srcIdx] : data[len - 1];
  }
  // Reflect right
  for (let i = 0; i < padRight; i++) {
    const srcIdx = len - 2 - i; // reflect index
    out[padLeft + len + i] = srcIdx >= 0 ? data[srcIdx] : data[0];
  }
  return out;
}

/**
 * Compute the spectrogram matching standalone_spec() from demucs ONNX export.
 * 
 * Returns: { data: Float32Array, totalChannels, freqBins, timeFrames }
 * Shape: [totalChannels, freqBins, timeFrames] ready for [1, ...] tensor
 */
function computeDemucsSpec(getChannelData, channels, length) {
  const hl = HOP_LENGTH;
  const le = Math.ceil(length / hl); // target number of output frames
  const pad = Math.floor(hl / 2) * 3; // 1536
  const rightPad = pad + le * hl - length;
  
  // Hann window
  const window = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
  }
  
  // After padding, the signal length is: pad + length + rightPad = 2*pad + le*hl
  // spectro() uses center=True which adds nfft//2 padding on each side
  // Total padded length for STFT: paddedLen + nfft
  // Number of STFT frames: floor(paddedSignalLen / hl) + 1 (with center=True)
  const paddedLen = pad + length + rightPad;
  const centerPad = Math.floor(N_FFT / 2);
  const totalLen = centerPad + paddedLen + centerPad;
  const totalFrames = Math.floor(paddedLen / hl) + 1;
  
  // After spectro: shape [C, n_fft//2+1, totalFrames] (complex)
  // Drop last freq bin: [C, n_fft//2, totalFrames]  (:-1 on freq axis)
  // Trim frames: [C, n_fft//2, le]  (2:2+le on time axis)
  const freqBins = Math.floor(N_FFT / 2); // 2048
  const timeFrames = le; // target output frames
  
  const totalChannels = channels * 2;
  const stftData = new Float32Array(totalChannels * freqBins * timeFrames);
  
  for (let c = 0; c < channels; c++) {
    const rawChannel = getChannelData(c);
    
    // Step 1: reflect pad the raw signal
    const paddedSignal = reflectPad(rawChannel, pad, rightPad);
    
    // Step 2: For each STFT frame (with center=True padding)
    for (let t = 0; t < totalFrames; t++) {
      // With center=True, frame starts at t*hl - centerPad in the padded signal
      const frameStart = t * hl - centerPad;
      
      const real = new Float32Array(N_FFT);
      const imag = new Float32Array(N_FFT);
      
      for (let i = 0; i < N_FFT; i++) {
        let idx = frameStart + i;
        // Reflect at boundaries of the padded signal
        if (idx < 0) idx = -idx;
        if (idx >= paddedLen) idx = 2 * paddedLen - idx - 2;
        idx = Math.max(0, Math.min(paddedLen - 1, idx));
        real[i] = paddedSignal[idx] * window[i];
      }
      
      fft(real, imag, N_FFT);
      
      // Normalize (PyTorch normalized=True)
      for (let f = 0; f < N_FFT; f++) {
        real[f] /= NORM_FACTOR;
        imag[f] /= NORM_FACTOR;
      }
      
      // Only store if this frame is in the trimmed range [2, 2+le)
      const outT = t - 2;
      if (outT >= 0 && outT < timeFrames) {
        const realChIdx = c * 2;
        const imagChIdx = c * 2 + 1;
        for (let f = 0; f < freqBins; f++) {
          // Bins 0..2047 (drop bin 2048 = Nyquist, i.e. [:-1] on freq axis)
          stftData[(realChIdx * freqBins + f) * timeFrames + outT] = real[f];
          stftData[(imagChIdx * freqBins + f) * timeFrames + outT] = imag[f];
        }
      }
    }
  }
  
  return { data: stftData, totalChannels, freqBins, timeFrames };
}

/**
 * In-place Cooley-Tukey FFT (radix-2, DIT)
 */
function fft(real, imag, n) {
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
  
  // Compute spectrogram matching standalone_spec + standalone_magnitude
  const getChannelData = (c) => inputData.subarray(c * chunkSize, (c + 1) * chunkSize);
  const { data: stftData, totalChannels, freqBins, timeFrames } = computeDemucsSpec(getChannelData, channels, chunkSize);
  const stftTensor = new ort.Tensor('float32', stftData, [1, totalChannels, freqBins, timeFrames]);
  
  console.warn(`  Chunk STFT shape: [1, ${totalChannels}, ${freqBins}, ${timeFrames}]`);
  
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
  
  // htdemucs_embedded: segment=10s at 44100Hz, but model expects 343980 samples
  const CHUNK_SIZE = 343980;
  const OVERLAP = Math.floor(CHUNK_SIZE * 0.25);
  const STEP = CHUNK_SIZE - OVERLAP;
  const NUM_STEMS = 4;
  
  const numChunks = Math.max(1, Math.ceil((length - OVERLAP) / STEP));
  
  console.warn(`Processing ${numChunks} chunk(s) of ${CHUNK_SIZE} samples (${(CHUNK_SIZE/sampleRate).toFixed(1)}s each)`);
  console.warn(`Total audio: ${(length/sampleRate).toFixed(1)}s, overlap: ${(OVERLAP/sampleRate).toFixed(1)}s`);
  
  const outputLength = Math.max(length, CHUNK_SIZE);
  const stemData = new Array(NUM_STEMS);
  const weightSum = new Float32Array(outputLength);
  for (let s = 0; s < NUM_STEMS; s++) {
    stemData[s] = new Array(channels);
    for (let c = 0; c < channels; c++) {
      stemData[s][c] = new Float32Array(outputLength);
    }
  }
  
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

/**
 * Demucs ONNX Model Loader
 * 
 * Port of sevagh/demucs.onnx C++ implementation to JavaScript.
 * 
 * STFT preprocessing (from dsp.cpp + model_inference.cpp):
 *   1. Standard center-padded STFT (pad = N_FFT/2 = 2048)
 *   2. nb_frames = n_samples/hop + 1, nb_bins = N_FFT/2 + 1 = 2049
 *   3. Periodic Hann window (L+1 points, drop last)
 *   4. Normalize by 1/sqrt(N_FFT) (PyTorch normalized=True)
 *   5. Drop last freq bin (2049→2048)
 *   6. Drop 2 time frames from each end (k+2 offset, dim-4 total)
 *   7. CaC layout: [ch0_real, ch0_imag, ch1_real, ch1_imag] x [freq] x [time]
 * 
 * Time-domain input is the raw (unpadded) audio signal.
 * 
 * Reference: https://github.com/sevagh/demucs.onnx
 */

import * as ort from 'onnxruntime-web';

const MODELS = {
  htdemucs: 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx',
};

const N_FFT = 4096;
const HOP = 1024;
const CENTER_PAD = N_FFT / 2; // 2048
const STFT_BINS = N_FFT / 2 + 1; // 2049
const MODEL_BINS = N_FFT / 2; // 2048 (drop Nyquist)
const NORM = 1.0 / Math.sqrt(N_FFT);

// Configure ONNX Runtime
if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
  ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 2);
} else {
  ort.env.wasm.numThreads = 1;
}
ort.env.wasm.simd = true;

/**
 * Periodic Hann window (matching C++ init_const_hann_window)
 * L+1 points, delete last one
 */
function periodicHann(size) {
  const w = new Float32Array(size);
  const floatN = size + 1;
  for (let n = 0; n < size; n++) {
    w[n] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * n / (floatN - 1)));
  }
  return w;
}

const WINDOW = periodicHann(N_FFT);

/**
 * In-place Cooley-Tukey FFT (radix-2)
 */
function fft(real, imag, n) {
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = -2 * Math.PI / len;
    const wR = Math.cos(ang), wI = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cR = 1, cI = 0;
      for (let k = 0; k < half; k++) {
        const e = i + k, o = e + half;
        const tR = cR * real[o] - cI * imag[o];
        const tI = cR * imag[o] + cI * real[o];
        real[o] = real[e] - tR; imag[o] = imag[e] - tI;
        real[e] += tR; imag[e] += tI;
        const nR = cR * wR - cI * wI;
        cI = cR * wI + cI * wR; cR = nR;
      }
    }
  }
}

/**
 * Compute STFT matching sevagh/demucs.onnx C++ implementation exactly.
 * 
 * Input: raw channel data (unpadded)
 * Output: CaC tensor data [channels*2, MODEL_BINS, trimmedFrames]
 */
function computeSTFT(getChannelData, channels, nSamples) {
  // Standard center padding: pad N_FFT/2 on each side
  const paddedLen = nSamples + N_FFT; // = nSamples + 2*CENTER_PAD
  const nbFrames = Math.floor(nSamples / HOP) + 1;
  
  // No trimming — htdemucs_embedded model expects all frames
  const outFrames = nbFrames;
  
  const totalCh = channels * 2;
  const data = new Float32Array(totalCh * MODEL_BINS * outFrames);
  
  for (let c = 0; c < channels; c++) {
    const raw = getChannelData(c);
    
    // Build padded signal with reflect padding (matching C++ pad_signal)
    const padded = new Float32Array(paddedLen);
    
    // Copy raw into middle (offset = CENTER_PAD)
    for (let i = 0; i < nSamples; i++) {
      padded[CENTER_PAD + i] = raw[i];
    }
    
    // Reflect pad start: copy from [CENTER_PAD+1 .. CENTER_PAD+CENTER_PAD] reversed
    for (let i = 0; i < CENTER_PAD; i++) {
      padded[i] = padded[2 * CENTER_PAD - i];
    }
    
    // Reflect pad end: copy from [paddedLen-CENTER_PAD-2 .. paddedLen-2*CENTER_PAD-1] reversed
    for (let i = 0; i < CENTER_PAD; i++) {
      padded[paddedLen - CENTER_PAD + i] = padded[paddedLen - CENTER_PAD - 2 - i];
    }
    
    // STFT: slide window with hop
    for (let frame = 0; frame < nbFrames; frame++) {
      const start = frame * HOP;
      
      const outFrame = frame;
      
      const real = new Float32Array(N_FFT);
      const imag = new Float32Array(N_FFT);
      
      // Window and copy
      for (let i = 0; i < N_FFT; i++) {
        real[i] = padded[start + i] * WINDOW[i];
      }
      
      fft(real, imag, N_FFT);
      
      // Store bins 0..2047 (drop bin 2048 = Nyquist), normalized
      // CaC layout: [ch0_real, ch0_imag, ch1_real, ch1_imag]
      const rCh = c * 2;
      const iCh = c * 2 + 1;
      for (let f = 0; f < MODEL_BINS; f++) {
        data[(rCh * MODEL_BINS + f) * outFrames + outFrame] = real[f] * NORM;
        data[(iCh * MODEL_BINS + f) * outFrames + outFrame] = imag[f] * NORM;
      }
    }
  }
  
  return { data, totalCh, outFrames };
}

// Load model
export async function loadDemucsModel(progressCallback) {
  progressCallback?.({ stage: 'downloading', percent: 0 });
  
  const response = await fetch(MODELS.htdemucs, { mode: 'cors' });
  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
  }
  const total = parseInt(response.headers.get('content-length'), 10) || 0;
  
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
      received, total
    });
  }
  
  const buf = new Uint8Array(received);
  let pos = 0;
  for (const chunk of chunks) { buf.set(chunk, pos); pos += chunk.length; }
  
  progressCallback?.({ stage: 'loading', percent: 0 });
  
  const session = await ort.InferenceSession.create(buf.buffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  
  progressCallback?.({ stage: 'ready', percent: 100 });
  console.warn('Demucs model loaded!');
  console.warn('Input names:', session.inputNames);
  console.warn('Output names:', session.outputNames);
  return session;
}

// Process a single chunk
async function processChunk(session, audioData, channels, offset, chunkSize) {
  const length = audioData.length;
  
  // Raw audio [1, channels, chunkSize] — unpadded, matching C++ xt input
  const inputData = new Float32Array(channels * chunkSize);
  for (let c = 0; c < channels; c++) {
    const ch = audioData.getChannelData(c);
    for (let i = 0; i < chunkSize; i++) {
      const idx = offset + i;
      if (idx >= 0 && idx < length) inputData[c * chunkSize + i] = ch[idx];
    }
  }
  
  const inputTensor = new ort.Tensor('float32', inputData, [1, channels, chunkSize]);
  
  // Compute STFT
  const getCh = (c) => inputData.subarray(c * chunkSize, (c + 1) * chunkSize);
  const { data: stftData, totalCh, outFrames } = computeSTFT(getCh, channels, chunkSize);
  const stftTensor = new ort.Tensor('float32', stftData, [1, totalCh, MODEL_BINS, outFrames]);
  
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  feeds[session.inputNames[1]] = stftTensor;
  
  const results = await session.run(feeds);
  return results[session.outputNames[1]]; // time-domain output
}

// Chunked processing
export async function separateStems(session, audioData, sampleRate, onProgress) {
  const channels = audioData.numberOfChannels;
  const length = audioData.length;
  const CHUNK = 343980;
  const OVERLAP = Math.floor(CHUNK * 0.25);
  const STEP = CHUNK - OVERLAP;
  const STEMS = 4;
  
  const numChunks = Math.max(1, Math.ceil((length - OVERLAP) / STEP));
  console.warn(`${numChunks} chunks, ${(length/sampleRate).toFixed(1)}s total`);
  
  const outLen = Math.max(length, CHUNK);
  const stemData = Array.from({length: STEMS}, () => 
    Array.from({length: channels}, () => new Float32Array(outLen))
  );
  const wSum = new Float32Array(outLen);
  
  const fade = new Float32Array(CHUNK);
  for (let i = 0; i < CHUNK; i++) {
    if (i < OVERLAP) fade[i] = i / OVERLAP;
    else if (i >= CHUNK - OVERLAP) fade[i] = (CHUNK - i) / OVERLAP;
    else fade[i] = 1;
  }
  
  for (let ci = 0; ci < numChunks; ci++) {
    const off = ci * STEP;
    const pct = Math.round(((ci + 1) / numChunks) * 100);
    console.warn(`Chunk ${ci+1}/${numChunks} (${pct}%)`);
    onProgress?.({ chunk: ci+1, total: numChunks, percent: pct });
    
    const out = await processChunk(session, audioData, channels, off, CHUNK);
    const d = out.data;
    const [, stems, ch, samples] = out.dims;
    
    for (let s = 0; s < stems; s++) {
      for (let c = 0; c < ch; c++) {
        for (let i = 0; i < samples; i++) {
          const oi = off + i;
          if (oi < outLen) {
            let w = fade[i];
            if (ci === 0 && i < OVERLAP) w = 1;
            if (ci === numChunks-1 && i >= CHUNK-OVERLAP) w = 1;
            stemData[s][c][oi] += d[s*ch*samples + c*samples + i] * w;
            if (s === 0 && c === 0) wSum[oi] += w;
          }
        }
      }
    }
  }
  
  for (let s = 0; s < STEMS; s++)
    for (let c = 0; c < channels; c++)
      for (let i = 0; i < length; i++)
        if (wSum[i] > 0) stemData[s][c][i] /= wSum[i];
  
  return { stemData, channels, length, numStems: STEMS };
}

// Extract to AudioBuffers
export function extractStems(result, sampleRate, audioContext) {
  const { stemData, channels, length, numStems } = result;
  const names = ['drums', 'bass', 'other', 'vocals'];
  const buffers = {};
  for (let s = 0; s < numStems; s++) {
    const buf = audioContext.createBuffer(channels, length, sampleRate);
    for (let c = 0; c < channels; c++)
      buf.getChannelData(c).set(stemData[s][c].subarray(0, length));
    buffers[names[s]] = buf;
  }
  return buffers;
}

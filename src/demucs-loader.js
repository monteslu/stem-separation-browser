/**
 * Demucs ONNX Model Loader with WASM DSP
 * 
 * Uses compiled C++ (Emscripten WASM) for STFT/iSTFT preprocessing,
 * matching the exact implementation from sevagh/demucs.onnx.
 * ONNX Runtime Web handles model inference.
 */

import * as ort from 'onnxruntime-web';

const MODEL_URL = 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx';
const MODEL_BINS = 2048;

if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
  ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 2);
} else {
  ort.env.wasm.numThreads = 1;
}
ort.env.wasm.simd = true;

let wasmModule = null;
let DemucsModuleFactory = null;

async function loadDemucsModuleFactory() {
  if (DemucsModuleFactory) return DemucsModuleFactory;
  const script = await import(/* @vite-ignore */ '/demucs_dsp.js');
  DemucsModuleFactory = script.default || globalThis.DemucsModule;
  return DemucsModuleFactory;
}

async function getWasm() {
  if (!wasmModule) {
    const factory = await loadDemucsModuleFactory();
    wasmModule = await factory({ locateFile: (path) => '/' + path });
  }
  return wasmModule;
}

async function computeSTFT(audioData, channels, nSamples) {
  const wasm = await getWasm();
  const bufSize = wasm.ccall('get_stft_buffer_size', 'number', ['number', 'number'], [nSamples, channels]);
  const nbFrames = wasm.ccall('get_stft_frames', 'number', ['number'], [nSamples]);
  
  const audioPtr = wasm._malloc(channels * nSamples * 4);
  const outPtr = wasm._malloc(bufSize * 4);
  const framesPtr = wasm._malloc(4);
  
  try {
    wasm.HEAPF32.set(audioData, audioPtr >> 2);
    const result = wasm.ccall('compute_stft', 'number',
      ['number', 'number', 'number', 'number', 'number'],
      [audioPtr, nSamples, channels, outPtr, framesPtr]);
    if (result !== 0) throw new Error('WASM compute_stft failed');
    
    const actualFrames = wasm.HEAP32[framesPtr >> 2];
    const totalCh = channels * 2;
    const dataLen = totalCh * MODEL_BINS * actualFrames;
    const data = new Float32Array(dataLen);
    data.set(wasm.HEAPF32.subarray(outPtr >> 2, (outPtr >> 2) + dataLen));
    return { data, totalCh, outFrames: actualFrames };
  } finally {
    wasm._free(audioPtr); wasm._free(outPtr); wasm._free(framesPtr);
  }
}

async function istftAndMerge(freqData, timeData, stems, channels, nSamples, nbFrames) {
  const wasm = await getWasm();
  const totalCh = channels * 2;
  const freqSize = stems * totalCh * MODEL_BINS * nbFrames;
  const timeSize = stems * channels * nSamples;
  
  const freqPtr = wasm._malloc(freqSize * 4);
  const timePtr = wasm._malloc(timeSize * 4);
  const outPtr = wasm._malloc(timeSize * 4);
  
  try {
    wasm.HEAPF32.set(freqData, freqPtr >> 2);
    wasm.HEAPF32.set(timeData, timePtr >> 2);
    const result = wasm.ccall('compute_istft_merge', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number'],
      [freqPtr, timePtr, stems, channels, nSamples, nbFrames, outPtr]);
    if (result !== 0) throw new Error('WASM compute_istft_merge failed');
    
    const data = new Float32Array(timeSize);
    data.set(wasm.HEAPF32.subarray(outPtr >> 2, (outPtr >> 2) + timeSize));
    return data;
  } finally {
    wasm._free(freqPtr); wasm._free(timePtr); wasm._free(outPtr);
  }
}

export async function loadDemucsModel(progressCallback) {
  const wasmPromise = getWasm();
  progressCallback?.({ stage: 'downloading', percent: 0 });
  
  const response = await fetch(MODEL_URL, { mode: 'cors' });
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  const total = parseInt(response.headers.get('content-length'), 10) || 0;
  
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    progressCallback?.({ stage: 'downloading', percent: total ? Math.round((received / total) * 100) : 0, received, total });
  }
  
  const buf = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { buf.set(c, pos); pos += c.length; }
  
  progressCallback?.({ stage: 'loading', percent: 0 });
  await wasmPromise;
  
  const session = await ort.InferenceSession.create(buf.buffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  
  progressCallback?.({ stage: 'ready', percent: 100 });
  console.warn('Demucs loaded! WASM DSP ready.');
  console.warn('Inputs:', session.inputNames, 'Outputs:', session.outputNames);
  return session;
}

async function processChunk(session, audioData, channels, offset, chunkSize) {
  const length = audioData.length;
  const inputData = new Float32Array(channels * chunkSize);
  for (let c = 0; c < channels; c++) {
    const ch = audioData.getChannelData(c);
    for (let i = 0; i < chunkSize; i++) {
      const idx = offset + i;
      if (idx >= 0 && idx < length) inputData[c * chunkSize + i] = ch[idx];
    }
  }
  
  const inputTensor = new ort.Tensor('float32', inputData, [1, channels, chunkSize]);
  const { data: stftData, totalCh, outFrames } = await computeSTFT(inputData, channels, chunkSize);
  const stftTensor = new ort.Tensor('float32', stftData, [1, totalCh, MODEL_BINS, outFrames]);
  
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  feeds[session.inputNames[1]] = stftTensor;
  
  const results = await session.run(feeds);
  const freqOutput = results[session.outputNames[0]];
  const timeOutput = results[session.outputNames[1]];
  const [, stems, ch, samples] = timeOutput.dims;
  
  const merged = await istftAndMerge(freqOutput.data, timeOutput.data, stems, ch, samples, outFrames);
  return { data: merged, dims: timeOutput.dims };
}

export async function separateStems(session, audioData, sampleRate, onProgress) {
  const channels = audioData.numberOfChannels;
  const length = audioData.length;
  const CHUNK = 343980, OVERLAP = Math.floor(CHUNK * 0.25), STEP = CHUNK - OVERLAP, STEMS = 4;
  const numChunks = Math.max(1, Math.ceil((length - OVERLAP) / STEP));
  console.warn(`${numChunks} chunks, ${(length/sampleRate).toFixed(1)}s total`);
  
  const outLen = Math.max(length, CHUNK);
  const stemData = Array.from({length: STEMS}, () => Array.from({length: channels}, () => new Float32Array(outLen)));
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
    
    const { data: d, dims } = await processChunk(session, audioData, channels, off, CHUNK);
    const [, stems, ch, samples] = dims;
    for (let s = 0; s < stems; s++)
      for (let c = 0; c < ch; c++)
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
  
  for (let s = 0; s < STEMS; s++)
    for (let c = 0; c < channels; c++)
      for (let i = 0; i < length; i++)
        if (wSum[i] > 0) stemData[s][c][i] /= wSum[i];
  
  return { stemData, channels, length, numStems: STEMS };
}

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

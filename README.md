# 🎵 Stem Separation POC

Browser-based audio stem separation proof of concept, researching feasibility for [Loukai](https://github.com/monteslu/loukai) karaoke app integration.

## Status: Research Phase 🔬

Currently exploring options for running audio source separation directly in the browser.

### The Challenge

Traditional stem separation models like **Demucs** and **Spleeter** are:
- PyTorch/TensorFlow-based
- Large (100MB+ model weights)
- Not directly compatible with browser runtimes

### Approaches Being Explored

| Approach | Pros | Cons | Status |
|----------|------|------|--------|
| **Transformers.js + ONNX** | Native browser, WebGPU support | Need ONNX-converted models | 🔍 Researching |
| **Demucs → ONNX conversion** | Best quality | Complex conversion, large models | 🔍 Researching |
| **Spleeter ONNX** | Lighter weight | May exist in community | 🔍 Searching |
| **Web Audio API (mid/side)** | Works now, no ML needed | Poor quality, stereo only | ✅ Demo mode |
| **WebAssembly (custom)** | Full control | Significant dev effort | ❌ Not started |

### Current Demo

The POC includes a fallback **mid/side separation** using Web Audio API:
- Extracts center-panned content (often vocals)
- Extracts side content (often instruments)
- Works with stereo audio only
- **Quality is poor** - this is just for demonstration

## Quick Start

```bash
npm install
npm run dev
```

Then open http://localhost:5173 and drop an audio file.

## Research Findings

### Transformers.js

[Transformers.js](https://huggingface.co/docs/transformers.js) is promising:
- Runs ONNX models in browser
- WebGPU acceleration available
- Audio capabilities exist (speech recognition, classification)
- **But**: No pre-converted source separation models found yet

### ONNX Runtime Web

The underlying runtime supports:
- WebGL backend (works everywhere)
- WebGPU backend (faster, newer browsers)
- WASM backend (good fallback)

### Model Conversion

Converting PyTorch models to ONNX is possible but challenging for audio models:
- Dynamic shapes (variable audio length)
- Complex architectures (U-Net style)
- Quality degradation risk

## 🎯 BREAKTHROUGH: Demucs ONNX Models Found!

Pre-converted Demucs ONNX models exist on HuggingFace:

| Model | Size | Notes |
|-------|------|-------|
| [timcsy/demucs-web-onnx](https://huggingface.co/timcsy/demucs-web-onnx) | ~180MB | htdemucs_embedded.onnx |
| [rysertio/Demucs-onnx](https://huggingface.co/rysertio/Demucs-onnx) | TBD | Music separation, quantized |
| [arjune123/demucs-onnx](https://huggingface.co/arjune123/demucs-onnx) | TBD | |
| [jackjiangxinfa/demucs-onnx](https://huggingface.co/jackjiangxinfa/demucs-onnx) | TBD | |

This is a major find - we can potentially use these pre-converted models directly with ONNX Runtime Web!

## Next Steps

1. [x] ~~Search HuggingFace Hub for ONNX audio separation models~~ FOUND!
2. [ ] Test loading `timcsy/demucs-web-onnx` with ONNX Runtime Web
3. [ ] Implement STFT preprocessing in JavaScript
4. [ ] Benchmark memory/performance with 180MB model
5. [ ] Consider WebGPU vs WASM backends

## Integration Path

If successful, integration into Loukai would:
1. Add stem separation as optional feature
2. Enable karaoke mode (mute vocals)
3. Support practice mode (isolate instruments)
4. Work offline after model download

## Contributing

This is a research POC. Findings will be reported to the Loukai project.

---

*Built with 🧙‍♂️ magic by Radagast*

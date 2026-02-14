/**
 * WASM API for Demucs STFT/iSTFT preprocessing.
 * 
 * Exposes functions for JavaScript to call:
 *   - compute_stft: raw audio -> CaC spectrogram tensor
 *   - compute_istft_and_merge: CaC spectrograms + time-domain stems -> final waveforms
 * 
 * Memory is managed via malloc/free from JS side.
 */

#include "dsp.hpp"
#include <cstdlib>
#include <cstring>

extern "C" {

/**
 * Compute STFT for stereo audio, output as CaC (complex-as-channels) tensor.
 * 
 * Input:
 *   audio_ptr: pointer to interleaved [ch0[nSamples], ch1[nSamples]] float32
 *   nSamples: number of samples per channel
 *   channels: number of channels (2 for stereo)
 * 
 * Output:
 *   out_ptr: pre-allocated buffer for [channels*2, MODEL_BINS, nbFrames] float32
 *   out_frames: pointer to int, receives nbFrames
 * 
 * Returns: 0 on success
 */
int compute_stft(
    const float* audio_ptr,
    int nSamples,
    int channels,
    float* out_ptr,
    int* out_frames)
{
    auto window = demucs::make_window();
    
    int nbFrames = 0;
    int totalCh = channels * 2;
    
    for (int c = 0; c < channels; c++) {
        const float* channelData = audio_ptr + c * nSamples;
        
        std::vector<std::vector<std::complex<float>>> spec;
        int frames;
        demucs::stft_channel(channelData, nSamples, window, spec, frames);
        
        if (c == 0) {
            nbFrames = frames;
            *out_frames = frames;
        }
        
        // Copy to CaC layout: [ch_real, ch_imag] interleaved per channel
        int rCh = c * 2;
        int iCh = c * 2 + 1;
        
        for (int f = 0; f < demucs::MODEL_BINS; f++) {
            for (int t = 0; t < nbFrames; t++) {
                out_ptr[(rCh * demucs::MODEL_BINS + f) * nbFrames + t] = spec[t][f].real();
                out_ptr[(iCh * demucs::MODEL_BINS + f) * nbFrames + t] = spec[t][f].imag();
            }
        }
    }
    
    return 0;
}

/**
 * Compute iSTFT on frequency-branch output and merge with time-branch output.
 * 
 * This does what model_inference.cpp does after RunONNXInference:
 *   result[source] = istft(freq_output[source]) + time_output[source]
 * 
 * Input:
 *   freq_ptr: CaC frequency output [stems, channels*2, MODEL_BINS, nbFrames] float32
 *   time_ptr: time-domain output [stems, channels, nSamples] float32
 *   stems: number of stems (4)
 *   channels: number of channels (2)
 *   nSamples: samples per channel per stem
 *   nbFrames: number of STFT frames
 * 
 * Output:
 *   out_ptr: pre-allocated [stems, channels, nSamples] float32
 */
int compute_istft_merge(
    const float* freq_ptr,
    const float* time_ptr,
    int stems,
    int channels,
    int nSamples,
    int nbFrames,
    float* out_ptr)
{
    auto window = demucs::make_window();
    int totalCh = channels * 2;
    
    for (int s = 0; s < stems; s++) {
        for (int c = 0; c < channels; c++) {
            // Extract complex spectrogram for this stem/channel from CaC
            int rCh = c * 2;
            int iCh = c * 2 + 1;
            
            std::vector<std::vector<std::complex<float>>> spec(nbFrames);
            for (int t = 0; t < nbFrames; t++) {
                spec[t].resize(demucs::STFT_BINS);
                for (int f = 0; f < demucs::MODEL_BINS; f++) {
                    int rIdx = s * totalCh * demucs::MODEL_BINS * nbFrames
                             + rCh * demucs::MODEL_BINS * nbFrames
                             + f * nbFrames + t;
                    int iIdx = s * totalCh * demucs::MODEL_BINS * nbFrames
                             + iCh * demucs::MODEL_BINS * nbFrames
                             + f * nbFrames + t;
                    spec[t][f] = std::complex<float>(freq_ptr[rIdx], freq_ptr[iIdx]);
                }
                // Last bin (Nyquist) = 0
                spec[t][demucs::MODEL_BINS] = std::complex<float>(0, 0);
            }
            
            // iSTFT
            std::vector<float> waveform(nSamples, 0.0f);
            demucs::istft_channel(spec, nbFrames, nSamples, window, waveform.data());
            
            // Merge: freq branch + time branch
            int outOffset = s * channels * nSamples + c * nSamples;
            int timeOffset = s * channels * nSamples + c * nSamples;
            
            for (int i = 0; i < nSamples; i++) {
                out_ptr[outOffset + i] = waveform[i] + time_ptr[timeOffset + i];
            }
        }
    }
    
    return 0;
}

/**
 * Get the number of STFT frames for a given sample count
 */
int get_stft_frames(int nSamples) {
    return nSamples / demucs::HOP + 1;
}

/**
 * Get required buffer size for STFT output
 */
int get_stft_buffer_size(int nSamples, int channels) {
    int nbFrames = nSamples / demucs::HOP + 1;
    return channels * 2 * demucs::MODEL_BINS * nbFrames;
}

} // extern "C"

#include "dsp.hpp"
#include <cstdlib>
#include <cstring>

extern "C" {

int compute_stft(const float* audio_ptr, int nSamples, int channels,
    float* out_ptr, int* out_frames)
{
    auto window = demucs::make_window();
    int nbFrames = 0;
    int totalCh = channels * 2;
    
    for (int c = 0; c < channels; c++) {
        const float* channelData = audio_ptr + c * nSamples;
        std::vector<std::vector<std::complex<float>>> spec;
        int frames;
        demucs::stft_channel(channelData, nSamples, window, spec, frames);
        if (c == 0) { nbFrames = frames; *out_frames = frames; }
        
        int rCh = c * 2, iCh = c * 2 + 1;
        for (int f = 0; f < demucs::MODEL_BINS; f++)
            for (int t = 0; t < nbFrames; t++) {
                out_ptr[(rCh * demucs::MODEL_BINS + f) * nbFrames + t] = spec[t][f].real();
                out_ptr[(iCh * demucs::MODEL_BINS + f) * nbFrames + t] = spec[t][f].imag();
            }
    }
    return 0;
}

int compute_istft_merge(const float* freq_ptr, const float* time_ptr,
    int stems, int channels, int nSamples, int nbFrames, float* out_ptr)
{
    auto window = demucs::make_window();
    int totalCh = channels * 2;
    
    for (int s = 0; s < stems; s++) {
        for (int c = 0; c < channels; c++) {
            int rCh = c * 2, iCh = c * 2 + 1;
            std::vector<std::vector<std::complex<float>>> spec(nbFrames);
            for (int t = 0; t < nbFrames; t++) {
                spec[t].resize(demucs::STFT_BINS);
                for (int f = 0; f < demucs::MODEL_BINS; f++) {
                    int rIdx = s*totalCh*demucs::MODEL_BINS*nbFrames + rCh*demucs::MODEL_BINS*nbFrames + f*nbFrames + t;
                    int iIdx = s*totalCh*demucs::MODEL_BINS*nbFrames + iCh*demucs::MODEL_BINS*nbFrames + f*nbFrames + t;
                    spec[t][f] = std::complex<float>(freq_ptr[rIdx], freq_ptr[iIdx]);
                }
                spec[t][demucs::MODEL_BINS] = std::complex<float>(0, 0);
            }
            
            std::vector<float> waveform(nSamples, 0.0f);
            demucs::istft_channel(spec, nbFrames, nSamples, window, waveform.data());
            
            int off = s * channels * nSamples + c * nSamples;
            for (int i = 0; i < nSamples; i++)
                out_ptr[off + i] = waveform[i] + time_ptr[off + i];
        }
    }
    return 0;
}

int get_stft_frames(int nSamples) { return nSamples / demucs::HOP + 1; }
int get_stft_buffer_size(int nSamples, int channels) {
    return channels * 2 * demucs::MODEL_BINS * (nSamples / demucs::HOP + 1);
}

}

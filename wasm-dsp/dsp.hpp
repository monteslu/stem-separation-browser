#ifndef DSP_HPP
#define DSP_HPP

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

namespace demucs {

constexpr int N_FFT = 4096;
constexpr int HOP = 1024;
constexpr int CENTER_PAD = N_FFT / 2;  // 2048
constexpr int STFT_BINS = N_FFT / 2 + 1;  // 2049
constexpr int MODEL_BINS = N_FFT / 2;  // 2048

// Periodic Hann window (L+1 points, drop last)
inline std::vector<float> make_window() {
    std::vector<float> w(N_FFT);
    float N = static_cast<float>(N_FFT + 1);
    for (int i = 0; i < N_FFT; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N - 1.0f)));
    }
    return w;
}

// Simple radix-2 FFT
void fft(std::vector<std::complex<float>>& data, int n);

// Inverse FFT
void ifft(std::vector<std::complex<float>>& data, int n);

// Compute STFT for one channel
// Input: raw samples (nSamples)
// Output: complex spectrogram [STFT_BINS x nbFrames]
void stft_channel(
    const float* input, int nSamples,
    const std::vector<float>& window,
    std::vector<std::vector<std::complex<float>>>& output,
    int& nbFrames
);

// Compute iSTFT for one channel
// Input: complex spectrogram [STFT_BINS x nbFrames]
// Output: waveform samples
void istft_channel(
    const std::vector<std::vector<std::complex<float>>>& spec,
    int nbFrames, int nSamples,
    const std::vector<float>& window,
    float* output
);

} // namespace demucs

#endif

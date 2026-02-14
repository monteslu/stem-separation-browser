#include "dsp.hpp"
#include <cstring>

namespace demucs {

void fft(std::vector<std::complex<float>>& data, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) std::swap(data[i], data[j]);
        int k = n >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        std::complex<float> wn(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1, 0);
            for (int k = 0; k < len / 2; k++) {
                auto t = w * data[i + k + len / 2];
                data[i + k + len / 2] = data[i + k] - t;
                data[i + k] += t;
                w *= wn;
            }
        }
    }
}

void ifft(std::vector<std::complex<float>>& data, int n) {
    for (int i = 0; i < n; i++) data[i] = std::conj(data[i]);
    fft(data, n);
    for (int i = 0; i < n; i++) data[i] = std::conj(data[i]) / static_cast<float>(n);
}

void stft_channel(const float* input, int nSamples, const std::vector<float>& window,
    std::vector<std::vector<std::complex<float>>>& output, int& nbFrames)
{
    int paddedLen = nSamples + N_FFT;
    nbFrames = nSamples / HOP + 1;
    
    std::vector<float> padded(paddedLen, 0.0f);
    std::memcpy(&padded[CENTER_PAD], input, nSamples * sizeof(float));
    
    // Reflect pad start
    for (int i = 0; i < CENTER_PAD; i++)
        padded[i] = padded[2 * CENTER_PAD - i];
    // Reflect pad end
    for (int i = 0; i < CENTER_PAD; i++)
        padded[paddedLen - CENTER_PAD + i] = padded[paddedLen - CENTER_PAD - 2 - i];
    
    output.resize(nbFrames);
    for (int i = 0; i < nbFrames; i++) output[i].resize(STFT_BINS);
    
    float norm = 1.0f / std::sqrt(static_cast<float>(N_FFT));
    std::vector<std::complex<float>> frame(N_FFT);
    
    for (int t = 0; t < nbFrames; t++) {
        int start = t * HOP;
        for (int i = 0; i < N_FFT; i++)
            frame[i] = std::complex<float>(padded[start + i] * window[i], 0.0f);
        fft(frame, N_FFT);
        for (int f = 0; f < STFT_BINS; f++)
            output[t][f] = frame[f] * norm;
    }
}

void istft_channel(const std::vector<std::vector<std::complex<float>>>& spec,
    int nbFrames, int nSamples, const std::vector<float>& window, float* output)
{
    int paddedLen = nSamples + N_FFT;
    std::vector<float> padded(paddedLen, 0.0f);
    std::vector<float> windowSum(paddedLen, 0.0f);
    float norm = std::sqrt(static_cast<float>(N_FFT));
    std::vector<std::complex<float>> frame(N_FFT);
    
    for (int t = 0; t < nbFrames; t++) {
        int start = t * HOP;
        for (int f = 0; f < STFT_BINS; f++)
            frame[f] = spec[t][f] * norm;
        for (int f = 1; f < N_FFT / 2; f++)
            frame[N_FFT - f] = std::conj(frame[f]);
        ifft(frame, N_FFT);
        for (int i = 0; i < N_FFT; i++) {
            padded[start + i] += frame[i].real() * window[i];
            windowSum[start + i] += window[i] * window[i];
        }
    }
    for (int i = 0; i < nSamples; i++) {
        float ws = windowSum[CENTER_PAD + i];
        output[i] = (ws > 1e-8f) ? padded[CENTER_PAD + i] / ws : 0.0f;
    }
}

}

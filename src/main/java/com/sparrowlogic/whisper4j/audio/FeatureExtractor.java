package com.sparrowlogic.whisper4j.audio;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.util.logging.Logger;

/**
 * Computes log-mel spectrogram features from raw audio, matching OpenAI Whisper's
 * feature extraction (ported from faster-whisper's FeatureExtractor).
 */
@SuppressWarnings({"checkstyle:ExecutableStatementCount", "checkstyle:MethodCount"})
public final class FeatureExtractor {

    private static final Logger LOG = Logger.getLogger(FeatureExtractor.class.getName());

    private final int nFft;
    private final int hopLength;
    private final int chunkLength;
    private final int samplingRate;
    private final int nMels;
    private final float[][] melFilters;

    public FeatureExtractor(final int nMels) {
        this(nMels, 16000, 160, 30, 400);
    }

    public FeatureExtractor(final int nMels, final int samplingRate, final int hopLength,
                            final int chunkLength, final int nFft) {
        this.nMels = nMels;
        this.samplingRate = samplingRate;
        this.hopLength = hopLength;
        this.chunkLength = chunkLength;
        this.nFft = nFft;
        this.melFilters = computeMelFilters(samplingRate, nFft, nMels);
    }

    /** Use pre-computed mel filters (e.g. from GGML model file). */
    public FeatureExtractor(final int nMels, final float[] melFilterData, final int nFftBins) {
        this.nMels = nMels;
        this.samplingRate = 16000;
        this.hopLength = 160;
        this.chunkLength = 30;
        this.nFft = (nFftBins - 1) * 2; // nFftBins = nFft/2 + 1, so nFft = (nFftBins-1)*2
        this.melFilters = new float[nMels][nFftBins];
        for (int m = 0; m < nMels; m++) {
            System.arraycopy(melFilterData, m * nFftBins, this.melFilters[m], 0, nFftBins);
        }
    }

    public int nSamples() {
        return this.chunkLength * this.samplingRate;
    }

    public int maxFrames() {
        return this.nSamples() / this.hopLength;
    }

    public int nMels() {
        return this.nMels;
    }

    /**
     * Compute log-mel spectrogram. Returns Tensor of shape (nMels, frames).
     */
    @SuppressWarnings("checkstyle:NestedForDepth")
    public Tensor extract(final float[] audio) {
        float[] input = audio;
        LOG.fine("Extracting features: %d samples (%.2fs at %dHz), %d mels"
                .formatted(input.length, input.length / (float) this.samplingRate,
                        this.samplingRate, this.nMels));
        // pad to at least nFft
        if (input.length < this.nFft) {
            float[] padded = new float[this.nFft];
            System.arraycopy(input, 0, padded, 0, input.length);
            input = padded;
        }

        // Zero-pad by nFft/2 on each side (matching whisper's STFT)
        int pad = this.nFft / 2;
        float[] padded = new float[input.length + 2 * pad];
        System.arraycopy(input, 0, padded, pad, input.length);

        // STFT
        float[] window = hannWindow(this.nFft);
        int nFrames = 1 + (padded.length - this.nFft) / this.hopLength;
        int fftBins = this.nFft / 2 + 1;

        // magnitudes squared: (fftBins, nFrames)
        float[][] magnitudes = new float[fftBins][nFrames];

        for (int t = 0; t < nFrames; t++) {
            // windowed frame
            float[] re = new float[this.nFft];
            float[] im = new float[this.nFft];
            for (int i = 0; i < this.nFft; i++) {
                re[i] = padded[t * this.hopLength + i] * window[i];
            }
            fft(re, im);
            for (int f = 0; f < fftBins; f++) {
                magnitudes[f][t] = re[f] * re[f] + im[f] * im[f];
            }
        }

        // mel spectrogram: (nMels, nFrames)
        float[] logSpec = new float[this.nMels * nFrames];
        float globalMax = Float.NEGATIVE_INFINITY;

        for (int m = 0; m < this.nMels; m++) {
            for (int t = 0; t < nFrames; t++) {
                float sum = 0;
                for (int f = 0; f < fftBins; f++) {
                    sum += this.melFilters[m][f] * magnitudes[f][t];
                }
                float val = (float) Math.log10(Math.max(sum, 1e-10));
                logSpec[m * nFrames + t] = val;
                if (val > globalMax) {
                    globalMax = val;
                }
            }
        }

        // clamp and normalize
        float floor = globalMax - 8.0f;
        for (int i = 0; i < logSpec.length; i++) {
            logSpec[i] = (Math.max(logSpec[i], floor) + 4.0f) / 4.0f;
        }

        LOG.fine("Features extracted: shape=(%d, %d)".formatted(this.nMels, nFrames));
        return Tensor.of(logSpec, this.nMels, nFrames);
    }

    /** Pad or trim features to maxFrames along the last dimension. Pads with mel floor value. */
    public Tensor padOrTrim(final Tensor features) {
        int frames = features.dim(-1);
        int target = this.maxFrames();
        if (frames == target) {
            return features;
        }
        // Find floor value (minimum in the mel spectrogram)
        float[] fData = features.data();
        float floor = fData[0];
        for (float v : fData) {
            if (v < floor) { floor = v; }
        }
        float[] out = new float[this.nMels * target];
        java.util.Arrays.fill(out, floor);
        int copyFrames = Math.min(frames, target);
        for (int m = 0; m < this.nMels; m++) {
            System.arraycopy(fData, m * frames, out, m * target, copyFrames);
        }
        return Tensor.of(out, this.nMels, target);
    }

    // ---- Mel filter bank (Slaney-style, matching faster-whisper) ----

    @SuppressWarnings("checkstyle:MethodName")
    static float[][] computeMelFilters(final int sr, final int nFft, final int nMels) {
        int fftBins = nFft / 2 + 1;
        float[] fftFreqs = new float[fftBins];
        for (int i = 0; i < fftBins; i++) {
            fftFreqs[i] = (float) i * sr / nFft;
        }

        float[] mels = new float[nMels + 2];
        float minMel = 0.0f;
        float maxMel = 45.245640471924965f;
        for (int i = 0; i <= nMels + 1; i++) {
            mels[i] = minMel + (maxMel - minMel) * i / (nMels + 1);
        }

        float[] freqs = new float[nMels + 2];
        float fMin = 0.0f;
        float fSp = 200.0f / 3.0f;
        float minLogHz = 1000.0f;
        float minLogMel = (minLogHz - fMin) / fSp;
        float logStep = (float) (Math.log(6.4) / 27.0);
        for (int i = 0; i < freqs.length; i++) {
            if (mels[i] < minLogMel) {
                freqs[i] = fMin + fSp * mels[i];
            } else {
                freqs[i] = minLogHz * (float) Math.exp(logStep * (mels[i] - minLogMel));
            }
        }

        float[][] weights = new float[nMels][fftBins];
        for (int m = 0; m < nMels; m++) {
            float fDiffLow = freqs[m + 1] - freqs[m];
            float fDiffHigh = freqs[m + 2] - freqs[m + 1];
            for (int f = 0; f < fftBins; f++) {
                float lower = (fftFreqs[f] - freqs[m]) / fDiffLow;
                float upper = (freqs[m + 2] - fftFreqs[f]) / fDiffHigh;
                weights[m][f] = Math.max(0, Math.min(lower, upper));
            }
            // Slaney normalization
            float enorm = 2.0f / (freqs[m + 2] - freqs[m]);
            for (int f = 0; f < fftBins; f++) {
                weights[m][f] *= enorm;
            }
        }
        return weights;
    }

    // ---- FFT / DFT ----

    /** Compute real FFT: returns (re, im) for bins 0..n/2. Radix-2 FFT or Bluestein for non-power-of-2. */
    @SuppressWarnings("checkstyle:ModifiedControlVariable")
    static void fft(final float[] re, final float[] im) {
        int n = re.length;
        if ((n & (n - 1)) == 0) {
            fftInPlace(re, im);
        } else {
            bluesteinFft(re, im, n);
        }
    }

    /** Bluestein's algorithm: converts arbitrary-size DFT to power-of-2 FFT via convolution. */
    private static void bluesteinFft(final float[] re, final float[] im, final int n) {
        int m = Integer.highestOneBit(2 * n - 1) << 1; // next power of 2 >= 2n-1

        // Chirp: w[k] = exp(-i*pi*k^2/n)
        float[] chirpRe = new float[n];
        float[] chirpIm = new float[n];
        for (int k = 0; k < n; k++) {
            double angle = -Math.PI * ((long) k * k % (2L * n)) / n;
            chirpRe[k] = (float) Math.cos(angle);
            chirpIm[k] = (float) Math.sin(angle);
        }

        // a[k] = x[k] * conj(chirp[k]), zero-padded to m
        float[] aRe = new float[m];
        float[] aIm = new float[m];
        for (int k = 0; k < n; k++) {
            aRe[k] = re[k] * chirpRe[k] + im[k] * chirpIm[k];
            aIm[k] = im[k] * chirpRe[k] - re[k] * chirpIm[k];
        }

        // b[k] = chirp[k] for k=0..n-1, chirp[n-k] for k=m-n+1..m-1
        float[] bRe = new float[m];
        float[] bIm = new float[m];
        bRe[0] = chirpRe[0];
        bIm[0] = chirpIm[0];
        for (int k = 1; k < n; k++) {
            bRe[k] = bRe[m - k] = chirpRe[k];
            bIm[k] = bIm[m - k] = chirpIm[k];
        }

        // Convolve via FFT: FFT(a), FFT(b), multiply, IFFT
        fftInPlace(aRe, aIm);
        fftInPlace(bRe, bIm);
        for (int k = 0; k < m; k++) {
            float tr = aRe[k] * bRe[k] - aIm[k] * bIm[k];
            float ti = aRe[k] * bIm[k] + aIm[k] * bRe[k];
            aRe[k] = tr;
            aIm[k] = ti;
        }
        // IFFT = conj(FFT(conj(x)))/m
        for (int k = 0; k < m; k++) { aIm[k] = -aIm[k]; }
        fftInPlace(aRe, aIm);
        float invM = 1.0f / m;
        for (int k = 0; k < m; k++) { aRe[k] *= invM; aIm[k] *= -invM; }

        // Result: X[k] = conj(chirp[k]) * conv[k]
        for (int k = 0; k < n; k++) {
            re[k] = aRe[k] * chirpRe[k] + aIm[k] * chirpIm[k];
            im[k] = aIm[k] * chirpRe[k] - aRe[k] * chirpIm[k];
        }
    }

    @SuppressWarnings({"checkstyle:ModifiedControlVariable", "checkstyle:NestedForDepth"})
    private static void fftInPlace(final float[] re, final float[] im) {
        int n = re.length;
        // bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            while ((j & bit) != 0) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j) {
                float tmp = re[i];
                re[i] = re[j];
                re[j] = tmp;
                tmp = im[i];
                im[i] = im[j];
                im[j] = tmp;
            }
        }
        // butterfly
        for (int len = 2; len <= n; len <<= 1) {
            double angle = -2.0 * Math.PI / len;
            float wRe = (float) Math.cos(angle);
            float wIm = (float) Math.sin(angle);
            for (int i = 0; i < n; i += len) {
                float curRe = 1;
                float curIm = 0;
                for (int j = 0; j < len / 2; j++) {
                    float uRe = re[i + j];
                    float uIm = im[i + j];
                    float vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
                    float vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
                    re[i + j] = uRe + vRe;
                    im[i + j] = uIm + vIm;
                    re[i + j + len / 2] = uRe - vRe;
                    im[i + j + len / 2] = uIm - vIm;
                    float newCurRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newCurRe;
                }
            }
        }
    }

    static float[] hannWindow(final int n) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            w[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / n));
        }
        return w;
    }
}

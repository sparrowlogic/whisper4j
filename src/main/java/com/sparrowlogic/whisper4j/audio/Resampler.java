package com.sparrowlogic.whisper4j.audio;

/**
 * Resamples float PCM audio and converts stereo to mono.
 * Linear interpolation, matching pots-voice Resampler pattern.
 */
public final class Resampler {

    public static final int WHISPER_SAMPLE_RATE = 16000;

    private Resampler() { }

    /** Convert multi-channel interleaved samples to mono by averaging. */
    public static float[] toMono(final float[] samples, final int channels) {
        if (channels == 1) {
            return samples;
        }
        int frames = samples.length / channels;
        float[] mono = new float[frames];
        for (int i = 0; i < frames; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += samples[i * channels + c];
            }
            mono[i] = sum / channels;
        }
        return mono;
    }

    /** Resample mono float PCM from srcRate to dstRate via linear interpolation. */
    public static float[] resample(final float[] samples, final int srcRate, final int dstRate) {
        if (srcRate == dstRate) {
            return samples;
        }
        int dstLen = (int) ((long) samples.length * dstRate / srcRate);
        float[] out = new float[dstLen];
        double ratio = (double) srcRate / dstRate;
        for (int i = 0; i < dstLen; i++) {
            double pos = i * ratio;
            int idx = (int) pos;
            float frac = (float) (pos - idx);
            float s0 = samples[idx];
            float s1 = (idx + 1 < samples.length) ? samples[idx + 1] : s0;
            out[i] = s0 + frac * (s1 - s0);
        }
        return out;
    }

    /** Convenience: convert to mono 16kHz from any format. */
    public static float[] toWhisperInput(final float[] samples, final int sampleRate,
                                         final int channels) {
        float[] mono = toMono(samples, channels);
        return resample(mono, sampleRate, WHISPER_SAMPLE_RATE);
    }
}

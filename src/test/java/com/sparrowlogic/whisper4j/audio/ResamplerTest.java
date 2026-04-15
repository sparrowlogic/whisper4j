package com.sparrowlogic.whisper4j.audio;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ResamplerTest {

    @Test
    void monoPassthrough() {
        float[] mono = {0.1f, 0.2f, 0.3f};
        assertSame(mono, Resampler.toMono(mono, 1));
    }

    @Test
    void stereoToMono() {
        float[] stereo = {0.2f, 0.4f, 0.6f, 0.8f};
        float[] mono = Resampler.toMono(stereo, 2);
        assertEquals(2, mono.length);
        assertEquals(0.3f, mono[0], 1e-6f);
        assertEquals(0.7f, mono[1], 1e-6f);
    }

    @Test
    void resampleSameRate() {
        float[] in = {1f, 2f, 3f};
        assertSame(in, Resampler.resample(in, 16000, 16000));
    }

    @Test
    void resampleDownsample() {
        // 4 samples at 4Hz → 2 samples at 2Hz
        float[] in = {0f, 1f, 2f, 3f};
        float[] out = Resampler.resample(in, 4, 2);
        assertEquals(2, out.length);
        assertEquals(0f, out[0], 1e-6f);
        assertEquals(2f, out[1], 1e-6f);
    }

    @Test
    void resampleUpsample() {
        float[] in = {0f, 4f};
        float[] out = Resampler.resample(in, 1, 2);
        assertEquals(4, out.length);
        assertEquals(0f, out[0], 1e-6f);
        assertEquals(2f, out[1], 1e-6f);
    }

    @Test
    void toWhisperInputStereo48k() {
        // 48kHz stereo → 16kHz mono
        float[] stereo48k = new float[48000 * 2]; // 1s stereo at 48kHz
        for (int i = 0; i < stereo48k.length; i++) {
            stereo48k[i] = 0.5f;
        }
        float[] result = Resampler.toWhisperInput(stereo48k, 48000, 2);
        assertEquals(16000, result.length);
        assertEquals(0.5f, result[0], 1e-6f);
    }
}

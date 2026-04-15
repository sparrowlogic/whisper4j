package com.sparrowlogic.whisper4j.audio;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FeatureExtractorTest {

    @Test
    void extractProducesCorrectShape() {
        var fe = new FeatureExtractor(80);
        // 1 second of 16kHz audio
        float[] audio = new float[16000];
        var features = fe.extract(audio);
        assertEquals(80, features.dim(0));
        assertTrue(features.dim(1) > 0);
    }

    @Test
    void extractWith128Mels() {
        var fe = new FeatureExtractor(128);
        float[] audio = new float[16000];
        var features = fe.extract(audio);
        assertEquals(128, features.dim(0));
    }

    @Test
    void padOrTrimPads() {
        var fe = new FeatureExtractor(80);
        float[] audio = new float[8000]; // 0.5s
        var features = fe.extract(audio);
        var padded = fe.padOrTrim(features);
        assertEquals(80, padded.dim(0));
        assertEquals(fe.maxFrames(), padded.dim(1));
    }

    @Test
    void padOrTrimTrims() {
        var fe = new FeatureExtractor(80);
        // 35 seconds — longer than 30s chunk
        float[] audio = new float[16000 * 35];
        var features = fe.extract(audio);
        var trimmed = fe.padOrTrim(features);
        assertEquals(fe.maxFrames(), trimmed.dim(1));
    }

    @Test
    void sineWaveProducesNonZeroFeatures() {
        var fe = new FeatureExtractor(80);
        float[] audio = new float[16000];
        // 440Hz sine wave
        for (int i = 0; i < audio.length; i++) {
            audio[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / 16000.0) * 0.5f;
        }
        var features = fe.extract(audio);
        // should have non-trivial energy in mel bins
        float max = Float.NEGATIVE_INFINITY;
        for (float v : features.data()) if (v > max) max = v;
        assertTrue(max > -1.0f, "sine wave should produce significant mel energy");
    }

    @Test
    void hannWindowSymmetric() {
        float[] w = FeatureExtractor.hannWindow(400);
        assertEquals(400, w.length);
        assertEquals(0, w[0], 1e-4, "window should start near zero");
        // symmetric
        for (int i = 0; i < 200; i++) {
            assertEquals(w[i], w[399 - i], 0.01, "window should be symmetric at " + i);
        }
    }

    @Test
    void fftOfSineHasPeakAtCorrectBin() {
        int n = 512;
        float[] re = new float[n], im = new float[n];
        int freq = 10; // 10 cycles in n samples
        for (int i = 0; i < n; i++) {
            re[i] = (float) Math.sin(2.0 * Math.PI * freq * i / n);
        }
        FeatureExtractor.fft(re, im);
        // peak should be at bin 10
        float maxMag = 0;
        int maxBin = 0;
        for (int i = 0; i < n / 2; i++) {
            float mag = re[i] * re[i] + im[i] * im[i];
            if (mag > maxMag) { maxMag = mag; maxBin = i; }
        }
        assertEquals(freq, maxBin);
    }

    @Test
    void melFilterBankShape() {
        float[][] filters = FeatureExtractor.computeMelFilters(16000, 400, 80);
        assertEquals(80, filters.length);
        assertEquals(201, filters[0].length); // nFft/2 + 1
    }
}

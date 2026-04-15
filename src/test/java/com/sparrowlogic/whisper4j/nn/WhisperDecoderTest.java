package com.sparrowlogic.whisper4j.nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WhisperDecoderTest {

    @Test
    void compressionRatioOfEmptyTokens() {
        assertEquals(0f, WhisperDecoder.compressionRatio(new int[0]));
    }

    @Test
    void compressionRatioOfRepeatedTokens() {
        // Repeated tokens should have high compression ratio
        int[] repeated = new int[100];
        for (int i = 0; i < 100; i++) {
            repeated[i] = 42;
        }
        float ratio = WhisperDecoder.compressionRatio(repeated);
        assertTrue(ratio > 2.0f, "Repeated tokens should compress well, got " + ratio);
    }

    @Test
    void compressionRatioOfRandomTokens() {
        // Random tokens should have low compression ratio
        int[] random = new int[100];
        for (int i = 0; i < 100; i++) {
            random[i] = i * 137 % 50000;
        }
        float ratio = WhisperDecoder.compressionRatio(random);
        assertTrue(ratio < 2.5f, "Random tokens should not compress much, got " + ratio);
    }

    @Test
    void decodeResultRecord() {
        var r = new WhisperDecoder.DecodeResult(
                new int[]{1, 2, 3}, -0.5f, 1.5f, 0.1f, 0.0f);
        assertArrayEquals(new int[]{1, 2, 3}, r.tokens());
        assertEquals(-0.5f, r.avgLogprob());
        assertEquals(1.5f, r.compressionRatio());
        assertEquals(0.1f, r.noSpeechProb());
        assertEquals(0.0f, r.temperature());
    }
}

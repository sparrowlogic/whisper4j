package com.sparrowlogic.whisper4j.audio;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class VoiceActivityDetectorTest {

    @Test
    void silenceProducesNoSegments() {
        var vad = new VoiceActivityDetector();
        float[] silence = new float[16000 * 5]; // 5s silence
        List<VoiceActivityDetector.SpeechSegment> segs = vad.detect(silence);
        assertTrue(segs.isEmpty());
    }

    @Test
    void loudAudioProducesOneSegment() {
        var vad = new VoiceActivityDetector();
        float[] loud = new float[16000 * 2]; // 2s
        for (int i = 0; i < loud.length; i++) {
            loud[i] = 0.5f * (float) Math.sin(2 * Math.PI * 440 * i / 16000.0);
        }
        List<VoiceActivityDetector.SpeechSegment> segs = vad.detect(loud);
        assertFalse(segs.isEmpty());
        assertEquals(0, segs.getFirst().startSample());
    }

    @Test
    void speechSegmentSeconds() {
        var seg = new VoiceActivityDetector.SpeechSegment(16000, 32000);
        assertEquals(1.0f, seg.startSeconds(16000), 1e-6f);
        assertEquals(2.0f, seg.endSeconds(16000), 1e-6f);
    }

    @Test
    void speechSilenceSpeechProducesTwoSegments() {
        // 0.5s speech, 3s silence, 0.5s speech at 16kHz
        var vad = new VoiceActivityDetector(0.01f, 250, 2000, 16000);
        float[] audio = new float[16000 * 4];
        // First 0.5s loud
        for (int i = 0; i < 8000; i++) {
            audio[i] = 0.5f * (float) Math.sin(2 * Math.PI * 440 * i / 16000.0);
        }
        // 3s silence (already zero)
        // Last 0.5s loud
        for (int i = 56000; i < 64000; i++) {
            audio[i] = 0.5f * (float) Math.sin(2 * Math.PI * 440 * i / 16000.0);
        }
        List<VoiceActivityDetector.SpeechSegment> segs = vad.detect(audio);
        assertEquals(2, segs.size());
    }

    @Test
    void shortSpeechBelowMinDurationFiltered() {
        // 100ms speech followed by 3s silence — segment should be closed and filtered (< 250ms min)
        var vad = new VoiceActivityDetector(0.01f, 250, 500, 16000);
        float[] audio = new float[16000 * 4]; // 4s
        for (int i = 0; i < 1600; i++) { // 100ms loud
            audio[i] = 0.5f;
        }
        // rest is silence — triggers segment closure after 500ms
        List<VoiceActivityDetector.SpeechSegment> segs = vad.detect(audio);
        assertTrue(segs.isEmpty(), "100ms speech should be filtered (min 250ms)");
    }
}

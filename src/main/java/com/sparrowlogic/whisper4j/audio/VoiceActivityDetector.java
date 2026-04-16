package com.sparrowlogic.whisper4j.audio;

import java.util.ArrayList;
import java.util.List;

/**
 * Energy-based Voice Activity Detection.
 * Splits audio into speech segments by detecting energy above a threshold.
 */
public final class VoiceActivityDetector {

    private final float threshold;
    private final int minSpeechSamples;
    private final int minSilenceSamples;
    private final int windowSamples;

    public VoiceActivityDetector() {
        this(0.01f, 250, 2000, 16000);
    }

    public VoiceActivityDetector(final float threshold, final int minSpeechMs,
                                 final int minSilenceMs, final int sampleRate) {
        this.threshold = threshold;
        this.minSpeechSamples = sampleRate * minSpeechMs / 1000;
        this.minSilenceSamples = sampleRate * minSilenceMs / 1000;
        this.windowSamples = sampleRate * 30 / 1000;
    }

    /**
     * Detect speech segments in 16 kHz mono audio using RMS energy in 30 ms windows.
     *
     * @param audio 16 kHz mono float PCM
     * @return list of speech segments with sample-level boundaries
     */
    public List<SpeechSegment> detect(final float[] audio) {
        // Compute per-window voiced flags
        int nWindows = (audio.length + this.windowSamples - 1) / this.windowSamples;
        boolean[] voiced = new boolean[nWindows];
        for (int w = 0; w < nWindows; w++) {
            int start = w * this.windowSamples;
            int end = Math.min(start + this.windowSamples, audio.length);
            voiced[w] = rms(audio, start, end) >= this.threshold;
        }

        // Merge voiced windows into segments with min speech/silence constraints
        return this.mergeWindows(voiced, audio.length);
    }

    private List<SpeechSegment> mergeWindows(final boolean[] voiced, final int audioLen) {
        List<SpeechSegment> segments = new ArrayList<>();
        int speechStart = -1;
        int silenceCount = 0;

        for (int w = 0; w < voiced.length; w++) {
            int pos = w * this.windowSamples;
            if (voiced[w]) {
                if (speechStart < 0) {
                    speechStart = pos;
                }
                silenceCount = 0;
            } else if (speechStart >= 0) {
                silenceCount += this.windowSamples;
                if (silenceCount >= this.minSilenceSamples) {
                    int speechEnd = pos - silenceCount + this.windowSamples;
                    this.addIfLongEnough(segments, speechStart, speechEnd);
                    speechStart = -1;
                    silenceCount = 0;
                }
            }
        }
        if (speechStart >= 0) {
            this.addIfLongEnough(segments, speechStart, audioLen);
        }
        return segments;
    }

    private void addIfLongEnough(final List<SpeechSegment> segments, final int start, final int end) {
        if (end - start >= this.minSpeechSamples) {
            segments.add(new SpeechSegment(start, end));
        }
    }

    private static float rms(final float[] audio, final int from, final int to) {
        double sum = 0;
        for (int i = from; i < to; i++) {
            sum += audio[i] * (double) audio[i];
        }
        return (float) Math.sqrt(sum / (to - from));
    }

    /** A speech segment with start/end sample indices. */
    public record SpeechSegment(int startSample, int endSample) {
        public float startSeconds(final int sr) {
            return this.startSample / (float) sr;
        }

        public float endSeconds(final int sr) {
            return this.endSample / (float) sr;
        }
    }
}

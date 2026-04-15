package com.sparrowlogic.whisper4j;

import java.nio.file.Path;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * File-based transcription test. Uses the tiny.en model for speed.
 * Expected outputs from faster-whisper test suite.
 */
public class TestTranscribeFiles {

    // Use tiny.en for fast iteration — switch to larger models once pipeline works
    static final String MODEL = "models/ggml-base.en.bin";
    static final String DATA = "src/test/resources/data/";

    public static void main(String[] args) throws Exception {
        Logger root = Logger.getLogger("com.sparrowlogic.whisper4j");
        root.setLevel(Level.FINE);
        root.setUseParentHandlers(false);
        ConsoleHandler h = new ConsoleHandler();
        h.setLevel(Level.FINE);
        root.addHandler(h);

        System.out.println("=== Loading model ===");
        long t0 = System.currentTimeMillis();
        WhisperModel model = WhisperModel.load(Path.of(MODEL));
        System.out.println("Model loaded in " + (System.currentTimeMillis() - t0) + " ms\n");

        // Test 1: stereo_diarization.wav (short, ~2s per channel)
        System.out.println("=== Test: stereo_diarization.wav ===");
        transcribeFile(model, DATA + "stereo_diarization.wav");
        // Expected (left channel): "He began a confused complaint against the wizard, who had vanished behind the curtain on the left."
        // Expected (right channel): "The horizon seems extremely distant."

        // Test 2: physicsworks.wav (~40s lecture)
        System.out.println("\n=== Test: physicsworks.wav ===");
        transcribeFile(model, DATA + "physicsworks.wav");
        // Expected first segment: "Now I want to return to the conservation of mechanical energy."
    }

    static void transcribeFile(WhisperModel model, String path) {
        try {
            long t0 = System.currentTimeMillis();
            System.out.println("Transcribing: " + path);
            model.transcribe(Path.of(path)).forEach(seg ->
                    System.out.printf("  [%.1f-%.1f] %s%n", seg.start(), seg.end(), seg.text())
            );
            System.out.println("Done in " + (System.currentTimeMillis() - t0) + " ms");
        } catch (Exception e) {
            System.err.println("FAILED: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

package com.sparrowlogic.whisper4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Tests loading and transcribing with non-GGML model formats.
 * Validates SafeTensors (HuggingFace), PyTorch (.pt), and ONNX loaders.
 */
public class TestModelFormats {

    static final String DATA = "src/test/resources/data/";
    static final String AUDIO = DATA + "stereo_diarization.wav";

    public static void main(String[] args) throws Exception {
        Logger root = Logger.getLogger("com.sparrowlogic.whisper4j");
        root.setLevel(Level.INFO);
        root.setUseParentHandlers(false);
        ConsoleHandler h = new ConsoleHandler();
        h.setLevel(Level.INFO);
        root.addHandler(h);

        // Test 1: HuggingFace SafeTensors directory
        testFormat("SafeTensors (HuggingFace dir)",
                Path.of("models/whisper-base.en-hf"));

        // Test 2: SafeTensors file directly
        testFormat("SafeTensors (file)",
                Path.of("models/whisper-base.en-hf/model.safetensors"));

        // Test 3: PyTorch (HuggingFace pytorch_model.bin)
        testFormat("PyTorch (HF pytorch_model.bin)",
                Path.of("models/whisper-base.en-hf/pytorch_model.bin"));

        // Test 4: ONNX split (encoder_model.onnx + decoder_model.onnx)
        testFormat("ONNX (split encoder/decoder)",
                Path.of("models/whisper-base.en-onnx"));

        // Test 5: GGML baseline for comparison
        Path ggml = Path.of("models/ggml-base.en.bin");
        if (Files.exists(ggml)) {
            testFormat("GGML (baseline)", ggml);
        }
    }

    static void testFormat(String label, Path modelPath) {
        System.out.println("\n=== " + label + " ===");
        if (!Files.exists(modelPath)) {
            System.out.println("  SKIPPED — model not found: " + modelPath);
            return;
        }
        try {
            long t0 = System.currentTimeMillis();
            WhisperModel model = WhisperModel.load(modelPath);
            long loadMs = System.currentTimeMillis() - t0;
            System.out.println("  Loaded in " + loadMs + " ms");

            t0 = System.currentTimeMillis();
            model.transcribe(Path.of(AUDIO)).forEach(seg ->
                    System.out.printf("  [%.1f-%.1f] %s%n", seg.start(), seg.end(), seg.text())
            );
            System.out.println("  Transcribed in " + (System.currentTimeMillis() - t0) + " ms");
            System.out.println("  PASS");
        } catch (Exception e) {
            System.out.println("  FAIL: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

package com.sparrowlogic.whisper4j;

import com.sparrowlogic.whisper4j.audio.FeatureExtractor;
import com.sparrowlogic.whisper4j.audio.Resampler;
import com.sparrowlogic.whisper4j.audio.WavParser;
import com.sparrowlogic.whisper4j.model.GgmlLoader;
import com.sparrowlogic.whisper4j.model.WeightStore;
import com.sparrowlogic.whisper4j.nn.Conv1d;
import com.sparrowlogic.whisper4j.nn.LayerNorm;
import com.sparrowlogic.whisper4j.nn.Linear;
import com.sparrowlogic.whisper4j.nn.ResidualAttentionBlock;
import com.sparrowlogic.whisper4j.nn.WhisperDecoder;
import com.sparrowlogic.whisper4j.nn.WhisperEncoder;
import com.sparrowlogic.whisper4j.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Stage-by-stage numerical validation against Python whisper reference.
 * Runs the Python script to generate ground truth, then compares Java output
 * at each pipeline stage within tolerance budgets.
 *
 * <p>Environment versions at time of authoring:
 * <ul>
 *   <li>whisper reference commit: cba3768</li>
 *   <li>Python: 3.14.3</li>
 *   <li>numpy: 2.4.3</li>
 *   <li>Java: OpenJDK 26 (Corretto)</li>
 * </ul>
 *
 * <p>Tolerance budgets (from ml-model-migration.md Phase 4):
 * <ul>
 *   <li>Single op (conv, linear): &lt; 0.001</li>
 *   <li>After encoder (6+ layers): &lt; 0.05</li>
 *   <li>Decoder logits: &lt; 0.5</li>
 *   <li>Top-K token IDs: exact match</li>
 * </ul>
 */
class ValidateAgainstReferenceTest {

    private static final Logger LOG = Logger.getLogger(ValidateAgainstReferenceTest.class.getName());
    private static final String MODELS_DIR =
            "models/";
    private static final String AUDIO =
            "src/test/resources/data/physicsworks.wav";
    private static final String PYTHON_SCRIPT = "tools/generate_reference_values.py";

    @ParameterizedTest(name = "validate stages: {0}")
    @ValueSource(strings = {
        "ggml-base.en.bin",
        "ggml-small.en.bin",
        "ggml-large-v3-turbo.bin"
    })
    void validateStagesAgainstPythonReference(final String modelFile) throws Exception {
        String modelPath = MODELS_DIR + modelFile;
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), modelFile + " not found");

        // Step 1: Run Python to get reference values
        LOG.info("Running Python reference for " + modelFile);
        Map<String, Object> ref = runPythonReference(modelPath);
        Map<String, Object> stages = asMap(ref.get("stages"));
        Map<String, Object> meta = asMap(ref.get("metadata"));
        LOG.info("Reference metadata: " + meta);

        // Step 2: Run Java pipeline
        LOG.info("Running Java pipeline for " + modelFile);
        var loader = new GgmlLoader();
        var weights = loader.load(Path.of(modelPath));
        var dims = loader.dimensions();

        // Mel spectrogram
        var wav = WavParser.parse(Path.of(AUDIO));
        float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        float[] audio30s = new float[30 * 16000];
        System.arraycopy(mono, 0, audio30s, 0, Math.min(mono.length, audio30s.length));
        var fe = new FeatureExtractor(dims.nMels(), loader.melFilters(), loader.melFilterNFft());
        Tensor mel = fe.extract(audio30s);
        mel = fe.padOrTrim(mel);
        assertStage("mel", mel.data(), stages, 0.01f);

        // Conv1 + GELU
        mel = mel.reshape(1, dims.nMels(), fe.maxFrames());
        var conv1 = new Conv1d(weights.get("encoder.conv1.weight"),
                weights.get("encoder.conv1.bias"), 1, 1);
        Tensor c1 = conv1.forward(mel).gelu();
        assertStage("conv1_gelu", c1.data(), stages, 0.001f);

        // Conv2 + GELU
        var conv2 = new Conv1d(weights.get("encoder.conv2.weight"),
                weights.get("encoder.conv2.bias"), 2, 1);
        Tensor c2 = conv2.forward(c1).gelu();
        assertStage("conv2_gelu", c2.data(), stages, 0.001f);

        // Transpose + positional embedding
        Tensor x = c2.transpose();
        x = x.add(weights.get("encoder.positional_embedding"));
        assertStage("pos_embed", x.data(), stages, 0.001f);

        // Encoder block 0
        var block0 = buildEncoderBlock(weights, "encoder.blocks.0.", dims.nAudioHead());
        x = block0.forward(x, null, null, null, "");
        assertStage("encoder_block0", x.data(), stages, 0.01f);

        // Full encoder (re-run from mel for clean state)
        var encoder = new WhisperEncoder(weights, dims);
        Tensor encOut = encoder.forward(mel);
        assertStage("encoder_output", encOut.data(), stages, 0.05f);

        // Decoder first step logits
        var model = WhisperModel.load(Path.of(modelPath));
        // Use reflection-free approach: just run full transcribe and check output
        // For stage validation, we test the encoder output matches, then verify
        // the decoder produces correct top-5 tokens via a fresh decode
        Map<String, Object> decRef = asMap(stages.get("decoder_logits"));
        int[] refTop5 = toIntArray(decRef.get("top5_tokens"));

        // Run decoder through the model's internal pipeline
        var decoder = new WhisperDecoder(weights, dims);
        int[] prompt = toIntArray(decRef.get("prompt"));
        Map<String, Tensor> kvCache = new HashMap<>();
        Tensor logits = decoder.forward(prompt, encOut, kvCache);
        float[] lastLogits = logits.getRow(logits.dim(1) - 1);
        assertStage("decoder_logits", lastLogits, stages, 0.5f);

        // Top-5 token IDs must match exactly
        int[] javaTop5 = topk(lastLogits, 5);
        LOG.info("Top-5 ref=" + java.util.Arrays.toString(refTop5)
                + " java=" + java.util.Arrays.toString(javaTop5));
        assertArrayEquals(refTop5, javaTop5,
                "Top-5 token IDs must match Python reference exactly");

        // Stage 8: Decoder step 1 — validate KV cache produces correct logits
        if (stages.containsKey("decoder_step1")) {
            Map<String, Object> step1Ref = asMap(stages.get("decoder_step1"));
            int token0 = refTop5[0];
            Tensor logits1 = decoder.forward(new int[]{token0}, encOut, kvCache);
            float[] lastLogits1 = logits1.getRow(logits1.dim(1) - 1);
            assertStage("decoder_step1", lastLogits1, stages, 0.5f);
            int[] refTop5Step1 = toIntArray(step1Ref.get("top5_tokens"));
            int[] javaTop5Step1 = topk(lastLogits1, 5);
            LOG.info("Step1 Top-5 ref=" + java.util.Arrays.toString(refTop5Step1)
                    + " java=" + java.util.Arrays.toString(javaTop5Step1));
            assertArrayEquals(refTop5Step1, javaTop5Step1,
                    "Step 1 top-5 tokens must match (validates KV cache correctness)");
        }

        LOG.info("ALL STAGES PASSED for " + modelFile);
    }

    /**
     * Multi-step decode parity: run 10 greedy steps with KV cache and compare
     * per-step logits against Python reference. Catches KV cache drift that
     * compounds across steps. Max allowed drift: 0.1 per step.
     */
    @ParameterizedTest(name = "validate decode drift: {0}")
    @ValueSource(strings = {"ggml-large-v3-turbo.bin"})
    void validateDecodeStepDrift(final String modelFile) throws Exception {
        String modelPath = MODELS_DIR + modelFile;
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), modelFile + " not found");

        // Run Python multi-step decode
        ProcessBuilder pb = new ProcessBuilder(
                "python3", "tools/validate_decode_steps.py", modelPath, AUDIO, "10");
        pb.redirectErrorStream(true);
        Process proc = pb.start();
        String output;
        try (var reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(proc.getInputStream()))) {
            output = reader.lines().collect(Collectors.joining("\n"));
        }
        int exit = proc.waitFor();
        if (exit != 0) { fail("Python decode steps failed:\n" + output); }
        Map<String, Object> ref = parseJson(output.trim());
        java.util.List<?> refSteps = (java.util.List<?>) ref.get("steps");

        // Run Java multi-step decode
        var loader = new GgmlLoader();
        var weights = loader.load(Path.of(modelPath));
        var dims = loader.dimensions();
        var wav = WavParser.parse(Path.of(AUDIO));
        float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        float[] audio30s = new float[30 * 16000];
        System.arraycopy(mono, 0, audio30s, 0, Math.min(mono.length, audio30s.length));
        var fe = new FeatureExtractor(dims.nMels(), loader.melFilters(), loader.melFilterNFft());
        Tensor mel = fe.padOrTrim(fe.extract(audio30s)).reshape(1, dims.nMels(), fe.maxFrames());
        var encoder = new WhisperEncoder(weights, dims);
        Tensor encOut = encoder.forward(mel);
        var decoder = new WhisperDecoder(weights, dims);

        // Get prompt from step 0 reference
        Map<String, Object> step0Ref = asMap(refSteps.getFirst());
        // Build prompt matching Python
        int nVocab = dims.nVocab();
        boolean isMulti = nVocab >= 51865;
        int dt = (nVocab - 51765 - (isMulti ? 1 : 0)) - 98;
        int sot = isMulti ? 50257 : 50256;
        java.util.List<Integer> promptList = new java.util.ArrayList<>();
        promptList.add(sot);
        if (isMulti) { promptList.add(50258); promptList.add(50358 + dt); }
        promptList.add(50362 + dt);
        int[] prompt = promptList.stream().mapToInt(Integer::intValue).toArray();

        Map<String, Tensor> kvCache = new HashMap<>();
        int[] cur = prompt;
        float maxDrift = 0;
        boolean allTokensMatch = true;

        for (int step = 0; step < refSteps.size(); step++) {
            Map<String, Object> stepRef = asMap(refSteps.get(step));
            int refToken = ((Number) stepRef.get("token")).intValue();
            double[] refFirst4 = toDoubleArray(stepRef.get("first4"));

            Tensor logits = decoder.forward(cur, encOut, kvCache);
            float[] last = logits.getRow(logits.dim(1) - 1);
            int javaToken = topk(last, 1)[0];

            // Measure drift on first 4 logits
            float stepMaxDrift = 0;
            for (int i = 0; i < Math.min(4, refFirst4.length); i++) {
                float diff = Math.abs(last[i] - (float) refFirst4[i]);
                stepMaxDrift = Math.max(stepMaxDrift, diff);
            }
            maxDrift = Math.max(maxDrift, stepMaxDrift);

            boolean tokenMatch = javaToken == refToken;
            if (!tokenMatch) { allTokensMatch = false; }

            LOG.info(String.format("Step %2d: ref=%5d java=%5d %s drift=%.4f",
                    step, refToken, javaToken, tokenMatch ? "✓" : "✗", stepMaxDrift));

            cur = new int[]{javaToken};
        }

        LOG.info(String.format("Max drift across all steps: %.4f", maxDrift));
        assertTrue(maxDrift < 0.03f,
                "KV cache drift must stay under 0.03 per step, was " + maxDrift);
        assertTrue(allTokensMatch,
                "All greedy tokens must match Python reference through " + refSteps.size() + " steps");
    }

    @Test
    void validateTranscriptionQuality() throws Exception {
        String modelPath = MODELS_DIR + "ggml-base.en.bin";
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), "base.en not found");
        WhisperModel model = WhisperModel.load(Path.of(modelPath));
        StringBuilder text = new StringBuilder();
        model.transcribe(Path.of(AUDIO)).forEach(s -> text.append(s.text()));
        String lower = text.toString().toLowerCase();
        assertTrue(lower.contains("energy"), "Should contain 'energy': " + lower);
        assertTrue(lower.contains("mechanical"), "Should contain 'mechanical': " + lower);
    }

    @Test
    void validateTimestampSegments() throws Exception {
        String modelPath = MODELS_DIR + "ggml-base.en.bin";
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), "base.en not found");
        WhisperModel model = WhisperModel.load(Path.of(modelPath));
        var wav = com.sparrowlogic.whisper4j.audio.WavParser.parse(Path.of(AUDIO));
        float[] mono = com.sparrowlogic.whisper4j.audio.Resampler.toWhisperInput(
                wav.samples(), wav.sampleRate(), wav.channels());
        // First 30s with timestamps enabled
        float[] clip = new float[16000 * 30];
        System.arraycopy(mono, 0, clip, 0, clip.length);
        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", 1, true, false, false);
        var segments = model.transcribe(clip, opts).toList();

        assertFalse(segments.isEmpty(), "Timestamps should produce segments");
        // First segment should start near 0
        assertTrue(segments.getFirst().start() < 2.0f,
                "First segment should start near 0, got " + segments.getFirst().start());
        // Segments should contain speech
        String text = segments.stream().map(WhisperModel.Segment::text)
                .collect(java.util.stream.Collectors.joining(" ")).toLowerCase();
        assertTrue(text.contains("energy") || text.contains("pendulum"),
                "Timestamp segments should contain speech: " + text);
        // Timestamps should be non-decreasing
        for (int i = 1; i < segments.size(); i++) {
            assertTrue(segments.get(i).start() >= segments.get(i - 1).start(),
                    "Timestamps should be non-decreasing at segment " + i);
        }
        LOG.info("Timestamp test: %d segments, first=[%.1f-%.1f]".formatted(
                segments.size(), segments.getFirst().start(), segments.getFirst().end()));
    }

    @Test
    void validateVadTranscription() throws Exception {
        String modelPath = MODELS_DIR + "ggml-base.en.bin";
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), "Model not found: " + modelPath);
        WhisperModel model = WhisperModel.load(Path.of(modelPath));
        var wav = WavParser.parse(Path.of(AUDIO));
        float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        float[] clip = new float[16000 * 30];
        System.arraycopy(mono, 0, clip, 0, clip.length);
        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", 1, false, true, false);
        var segments = model.transcribe(clip, opts).toList();
        assertFalse(segments.isEmpty(), "VAD should produce segments");
        String text = segments.stream().map(WhisperModel.Segment::text)
                .collect(java.util.stream.Collectors.joining(" ")).toLowerCase();
        assertTrue(text.contains("energy") || text.contains("conservation"),
                "VAD transcription should contain speech: " + text);
    }

    @Test
    void validateLanguageDetection() throws Exception {
        String modelPath = MODELS_DIR + "ggml-base.en.bin";
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), "Model not found: " + modelPath);
        WhisperModel model = WhisperModel.load(Path.of(modelPath));
        var wav = WavParser.parse(Path.of(AUDIO));
        float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        String lang = model.detectLanguage(mono);
        assertEquals("en", lang, "Should detect English");
    }

    @Test
    void validateDefaultTranscribe() throws Exception {
        String modelPath = MODELS_DIR + "ggml-base.en.bin";
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), "Model not found: " + modelPath);
        WhisperModel model = WhisperModel.load(Path.of(modelPath));
        // Default options: beam=5, timestamps=true
        var segments = model.transcribe(Path.of(AUDIO)).toList();
        assertFalse(segments.isEmpty(), "Default transcribe should produce segments");
    }

    @Test
    void validateWordTimestamps() throws Exception {
        String modelPath = MODELS_DIR + "ggml-base.en.bin";
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(Path.of(modelPath)), "Model not found: " + modelPath);
        WhisperModel model = WhisperModel.load(Path.of(modelPath));
        var wav = WavParser.parse(Path.of(AUDIO));
        float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        float[] clip = new float[16000 * 10];
        System.arraycopy(mono, 0, clip, 0, clip.length);
        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", 1, true, false, false);
        var segments = model.transcribe(clip, opts).toList();
        if (!segments.isEmpty()) {
            var words = WhisperModel.wordTimestamps(segments.getFirst());
            // Word timestamps may be empty if no cross-attention weights captured
            // but the method should not throw
            LOG.info("Word timestamps: " + words.size() + " words from segment");
        }
    }

    // ---- Helpers ----

    private Map<String, Object> runPythonReference(final String modelPath) throws Exception {
        ProcessBuilder pb = new ProcessBuilder(
                "python3", PYTHON_SCRIPT, modelPath, AUDIO);
        pb.redirectErrorStream(true);
        Process proc = pb.start();
        String output;
        try (var reader = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
            output = reader.lines().collect(Collectors.joining("\n"));
        }
        int exit = proc.waitFor();
        if (exit != 0) {
            fail("Python script failed (exit " + exit + "):\n" + output);
        }
        // Parse JSON manually (no Jackson dependency in test scope)
        return parseJson(output);
    }

    private void assertStage(final String name, final float[] javaValues,
                             final Map<String, Object> stages, final float tolerance) {
        Map<String, Object> stage = asMap(stages.get(name));
        if (stage == null) {
            fail("No reference data for stage: " + name);
        }
        double[] refFirst8 = toDoubleArray(stage.get("first8"));
        int n = Math.min(refFirst8.length, Math.min(8, javaValues.length));
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[%s] tolerance=%.4f%n", name, tolerance));
        boolean pass = true;
        for (int i = 0; i < n; i++) {
            float diff = Math.abs(javaValues[i] - (float) refFirst8[i]);
            String status = diff <= tolerance ? "OK" : "FAIL";
            if (diff > tolerance) { pass = false; }
            sb.append(String.format("  [%d] ref=%.6f java=%.6f diff=%.6f %s%n",
                    i, refFirst8[i], javaValues[i], diff, status));
        }
        LOG.info(sb.toString());
        assertTrue(pass, "Stage " + name + " exceeded tolerance " + tolerance);
    }

    private static int[] topk(final float[] values, final int k) {
        int[] idx = new int[k];
        float[] top = new float[k];
        java.util.Arrays.fill(top, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < values.length; i++) {
            if (values[i] > top[k - 1]) {
                int pos = k - 1;
                while (pos > 0 && values[i] > top[pos - 1]) { pos--; }
                System.arraycopy(top, pos, top, pos + 1, k - 1 - pos);
                System.arraycopy(idx, pos, idx, pos + 1, k - 1 - pos);
                top[pos] = values[i];
                idx[pos] = i;
            }
        }
        return idx;
    }

    private static ResidualAttentionBlock buildEncoderBlock(final WeightStore w,
                                                            final String bp, final int nHead) {
        return new ResidualAttentionBlock(
                WhisperEncoder.buildMHA(w, bp + "attn.", nHead),
                new LayerNorm(w.get(bp + "attn_ln.weight"), w.get(bp + "attn_ln.bias")),
                null, null,
                new Linear(w.get(bp + "mlp.0.weight"), w.get(bp + "mlp.0.bias")),
                new Linear(w.get(bp + "mlp.2.weight"), w.get(bp + "mlp.2.bias")),
                new LayerNorm(w.get(bp + "mlp_ln.weight"), w.get(bp + "mlp_ln.bias"))
        );
    }

    // ---- Minimal JSON parser (avoids Jackson dependency) ----

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseJson(final String json) {
        // Use the JDK's Nashorn-free approach: simple recursive descent
        // For robustness, delegate to a tiny state machine
        return (Map<String, Object>) new SimpleJsonParser(json.trim()).parseValue();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(final Object o) {
        return (Map<String, Object>) o;
    }

    private static double[] toDoubleArray(final Object o) {
        java.util.List<?> list = (java.util.List<?>) o;
        double[] arr = new double[list.size()];
        for (int i = 0; i < arr.length; i++) { arr[i] = ((Number) list.get(i)).doubleValue(); }
        return arr;
    }

    private static int[] toIntArray(final Object o) {
        java.util.List<?> list = (java.util.List<?>) o;
        int[] arr = new int[list.size()];
        for (int i = 0; i < arr.length; i++) { arr[i] = ((Number) list.get(i)).intValue(); }
        return arr;
    }

    /** Minimal JSON parser — handles objects, arrays, strings, numbers, booleans, null. */
    private static class SimpleJsonParser {
        private final String s;
        private int pos;

        SimpleJsonParser(final String s) { this.s = s; this.pos = 0; }

        Object parseValue() {
            skipWs();
            char c = s.charAt(pos);
            if (c == '{') { return parseObject(); }
            if (c == '[') { return parseArray(); }
            if (c == '"') { return parseString(); }
            if (c == 't' || c == 'f') { return parseBoolean(); }
            if (c == 'n') { pos += 4; return null; }
            return parseNumber();
        }

        Map<String, Object> parseObject() {
            Map<String, Object> map = new java.util.LinkedHashMap<>();
            pos++; skipWs();
            if (s.charAt(pos) == '}') { pos++; return map; }
            while (true) {
                skipWs(); String key = parseString(); skipWs();
                pos++; // ':'
                skipWs(); Object val = parseValue();
                map.put(key, val); skipWs();
                if (s.charAt(pos) == '}') { pos++; return map; }
                pos++; // ','
            }
        }

        java.util.List<Object> parseArray() {
            java.util.List<Object> list = new java.util.ArrayList<>();
            pos++; skipWs();
            if (s.charAt(pos) == ']') { pos++; return list; }
            while (true) {
                skipWs(); list.add(parseValue()); skipWs();
                if (s.charAt(pos) == ']') { pos++; return list; }
                pos++; // ','
            }
        }

        String parseString() {
            pos++; // opening "
            StringBuilder sb = new StringBuilder();
            while (s.charAt(pos) != '"') {
                if (s.charAt(pos) == '\\') { pos++; }
                sb.append(s.charAt(pos++));
            }
            pos++; // closing "
            return sb.toString();
        }

        Number parseNumber() {
            int start = pos;
            while (pos < s.length() && "0123456789.eE+-".indexOf(s.charAt(pos)) >= 0) { pos++; }
            String num = s.substring(start, pos);
            if (num.contains(".") || num.contains("e") || num.contains("E")) {
                return Double.parseDouble(num);
            }
            long v = Long.parseLong(num);
            return v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE ? (int) v : v;
        }

        boolean parseBoolean() {
            if (s.charAt(pos) == 't') { pos += 4; return true; }
            pos += 5; return false;
        }

        void skipWs() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) { pos++; }
        }
    }
}

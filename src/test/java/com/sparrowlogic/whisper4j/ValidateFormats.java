package com.sparrowlogic.whisper4j;

import com.sparrowlogic.whisper4j.audio.FeatureExtractor;
import com.sparrowlogic.whisper4j.audio.Resampler;
import com.sparrowlogic.whisper4j.audio.WavParser;
import com.sparrowlogic.whisper4j.model.GgmlLoader;
import com.sparrowlogic.whisper4j.model.ModelDimensions;
import com.sparrowlogic.whisper4j.model.ModelLoader;
import com.sparrowlogic.whisper4j.model.WeightStore;
import com.sparrowlogic.whisper4j.nn.Conv1d;
import com.sparrowlogic.whisper4j.nn.WhisperDecoder;
import com.sparrowlogic.whisper4j.nn.WhisperEncoder;
import com.sparrowlogic.whisper4j.tensor.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Validates all model formats produce identical stage-by-stage outputs.
 * GGML is the reference. Each format must match within tolerance at every stage:
 *   1. Conv1+GELU  2. Conv2+GELU  3. Encoder output  4. Decoder top-5 tokens
 *
 * This ensures weight loading, name normalization, and dtype conversion are correct.
 */
public class ValidateFormats {

    static final String AUDIO = "src/test/resources/data/stereo_diarization.wav";

    // Reference values from GGML base.en (ground truth)
    static final float[] REF_CONV1 = {0.03736455f, 0.00012353f, 0.02271019f, 0.05902079f};
    static final float[] REF_CONV2 = {-0.10205206f, -0.00445322f, -0.00545253f, -0.00548453f};
    static final float[] REF_ENCODER = {-0.49441636f, 0.29003060f, -1.24786484f, -0.09641895f};
    static final int[] REF_TOP5 = {50256, 685, 383, 679, 1635};
    static final float REF_EOT = 12.069f;

    // Tolerance: F16 dequantization introduces ~0.001 error, compounding across layers
    static final float TOL_CONV = 0.01f;
    static final float TOL_ENCODER = 0.05f;
    static final float TOL_LOGIT = 0.5f;

    public static void main(final String[] args) throws Exception {
        Map<String, Path> models = new LinkedHashMap<>();
        models.put("GGML", Path.of(
                "models/ggml-base.en.bin"));
        models.put("SafeTensors", Path.of("models/whisper-base.en-hf/model.safetensors"));
        models.put("SafeTensors (dir)", Path.of("models/whisper-base.en-hf"));
        models.put("PyTorch", Path.of("models/whisper-base.en-hf/pytorch_model.bin"));

        // Prepare audio once (shared across all formats)
        WavParser.WavData wav = WavParser.parse(Path.of(AUDIO));
        float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        float[] audio30s = new float[30 * 16000];
        System.arraycopy(mono, 0, audio30s, 0, Math.min(mono.length, audio30s.length));

        int passed = 0;
        int total = 0;
        for (var entry : models.entrySet()) {
            total++;
            if (!Files.exists(entry.getValue())) {
                System.out.printf("SKIP  %-20s — not found: %s%n", entry.getKey(), entry.getValue());
                continue;
            }
            boolean ok = validateFormat(entry.getKey(), entry.getValue(), audio30s);
            if (ok) {
                passed++;
            }
        }
        System.out.printf("%n=== %d/%d formats passed ===%n", passed, total);
        if (passed < total) {
            System.exit(1);
        }
    }

    @SuppressWarnings("checkstyle:MethodLength")
    static boolean validateFormat(final String label, final Path modelPath,
                                  final float[] audio30s) {
        System.out.printf("%n=== %s ===%n", label);
        try {
            // Load weights — use WhisperModel's resolution for directories
            long t0 = System.currentTimeMillis();
            Path weightsPath = resolveWeightsPath(modelPath);
            ModelLoader loader = ModelLoader.forPath(weightsPath);
            WeightStore weights = loader.load(weightsPath);

            ModelDimensions dims;
            FeatureExtractor fe;
            if (loader instanceof GgmlLoader ggml) {
                dims = ggml.dimensions();
                fe = new FeatureExtractor(dims.nMels(), ggml.melFilters(), ggml.melFilterNFft());
            } else {
                weights.normalizeNames();
                dims = ModelDimensions.infer(weights);
                fe = new FeatureExtractor(dims.nMels());
            }
            System.out.printf("  Loaded %d tensors in %d ms%n", weights.size(), System.currentTimeMillis() - t0);

            // Stage 1: Mel spectrogram (same for all formats — audio processing only)
            Tensor mel = fe.extract(audio30s);
            mel = fe.padOrTrim(mel);
            Tensor mel3d = mel.reshape(1, dims.nMels(), fe.maxFrames());

            // Stage 2: Conv1 + GELU
            Conv1d conv1 = new Conv1d(weights.get("encoder.conv1.weight"),
                    weights.get("encoder.conv1.bias"), 1, 1);
            Tensor c1 = conv1.forward(mel3d).gelu();
            boolean conv1Ok = checkFirst4("Conv1+GELU", c1, REF_CONV1, TOL_CONV);

            // Stage 3: Conv2 + GELU
            Conv1d conv2 = new Conv1d(weights.get("encoder.conv2.weight"),
                    weights.get("encoder.conv2.bias"), 2, 1);
            Tensor c2 = conv2.forward(c1).gelu();
            boolean conv2Ok = checkFirst4("Conv2+GELU", c2, REF_CONV2, TOL_CONV);

            // Stage 4: Full encoder
            WhisperEncoder enc = new WhisperEncoder(weights, dims);
            Tensor encOut = enc.forward(mel3d);
            boolean encOk = checkFirst4("Encoder", encOut, REF_ENCODER, TOL_ENCODER);

            // Stage 5: Decoder — check top-5 token IDs and EOT score
            WhisperDecoder dec = new WhisperDecoder(weights, dims);
            int[] prompt = {50257, 50258, 50359, 50363};
            Tensor logits = dec.forward(prompt, encOut, null);
            float[] ld = logits.data();
            int nVocab = logits.dim(2);
            int lastPos = logits.dim(1) - 1;
            float[] lastLogits = new float[nVocab];
            System.arraycopy(ld, lastPos * nVocab, lastLogits, 0, nVocab);

            int[] top5 = topK(lastLogits, 5);
            boolean top5Ok = Arrays.equals(top5, REF_TOP5);
            float eot = lastLogits[50256];
            boolean eotOk = Math.abs(eot - REF_EOT) < TOL_LOGIT;

            System.out.printf("  Decoder top5=%s %s (ref=%s)%n",
                    Arrays.toString(top5), top5Ok ? "PASS" : "FAIL", Arrays.toString(REF_TOP5));
            System.out.printf("  Decoder EOT=%.3f %s (ref=%.3f, tol=%.1f)%n",
                    eot, eotOk ? "PASS" : "FAIL", REF_EOT, TOL_LOGIT);

            boolean allOk = conv1Ok && conv2Ok && encOk && top5Ok && eotOk;
            System.out.printf("  %s %s%n", label, allOk ? "PASS" : "FAIL");
            return allOk;
        } catch (Exception e) {
            System.out.printf("  FAIL: %s%n", e.getMessage());
            e.printStackTrace(System.out);
            return false;
        }
    }

    static boolean checkFirst4(final String stage, final Tensor t,
                               final float[] ref, final float tol) {
        float[] d = t.data();
        float maxErr = 0;
        for (int i = 0; i < 4; i++) {
            maxErr = Math.max(maxErr, Math.abs(d[i] - ref[i]));
        }
        boolean ok = maxErr < tol;
        System.out.printf("  %-12s first4=[%.6f, %.6f, %.6f, %.6f] maxErr=%.6f %s%n",
                stage, d[0], d[1], d[2], d[3], maxErr, ok ? "PASS" : "FAIL");
        return ok;
    }

    static int[] topK(final float[] a, final int k) {
        int[] idx = new int[k];
        float[] vals = new float[k];
        Arrays.fill(vals, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < a.length; i++) {
            if (a[i] > vals[k - 1]) {
                vals[k - 1] = a[i];
                idx[k - 1] = i;
                for (int j = k - 2; j >= 0; j--) {
                    if (vals[j + 1] > vals[j]) {
                        float tv = vals[j]; vals[j] = vals[j + 1]; vals[j + 1] = tv;
                        int ti = idx[j]; idx[j] = idx[j + 1]; idx[j + 1] = ti;
                    }
                }
            }
        }
        return idx;
    }

    /** Same resolution logic as WhisperModel.resolveWeightsPath. */
    @SuppressWarnings("checkstyle:ReturnCount")
    static Path resolveWeightsPath(final Path path) {
        if (!Files.isDirectory(path)) {
            return path;
        }
        Path safetensors = path.resolve("model.safetensors");
        if (Files.exists(safetensors)) {
            return safetensors;
        }
        Path pytorch = path.resolve("pytorch_model.bin");
        if (Files.exists(pytorch)) {
            return pytorch;
        }
        Path onnx = path.resolve("model.onnx");
        if (Files.exists(onnx)) {
            return onnx;
        }
        Path pt = path.resolve("model.pt");
        if (Files.exists(pt)) {
            return pt;
        }
        return path;
    }
}

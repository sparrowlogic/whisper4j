package com.sparrowlogic.whisper4j;

import com.sparrowlogic.whisper4j.audio.*;
import com.sparrowlogic.whisper4j.model.*;
import com.sparrowlogic.whisper4j.nn.*;
import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.nio.file.Path;

public class ValidateStages {
    static final String MODEL = "models/ggml-base.en.bin";
    static final String AUDIO = "src/test/resources/data/stereo_diarization.wav";

    public static void main(String[] args) throws Exception {
        // Load model
        var loader = new GgmlLoader();
        var weights = loader.load(Path.of(MODEL));
        var dims = loader.dimensions();

        // Stage 1: Mel
        var wav = WavParser.parse(Path.of(AUDIO));
        float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        float[] audio30s = new float[30 * 16000];
        System.arraycopy(mono, 0, audio30s, 0, Math.min(mono.length, audio30s.length));
        var fe = new FeatureExtractor(dims.nMels(), loader.melFilters(), loader.melFilterNFft());
        Tensor mel = fe.extract(audio30s);
        mel = fe.padOrTrim(mel);
        float[] md = mel.data();
        System.out.printf("[Stage 1] Mel: shape=%s mean=%.6f first4=[%.8f, %.8f, %.8f, %.8f]%n",
                mel, mean(md), md[0], md[1], md[2], md[3]);

        // Stage 2: Conv1 + GELU
        Tensor mel3d = mel.reshape(1, dims.nMels(), fe.maxFrames());
        var conv1 = new Conv1d(weights.get("encoder.conv1.weight"), weights.get("encoder.conv1.bias"), 1, 1);
        Tensor c1 = conv1.forward(mel3d).gelu();
        float[] c1d = c1.data();
        System.out.printf("[Stage 2] Conv1+GELU: shape=%s first4=[%.8f, %.8f, %.8f, %.8f]%n",
                c1, c1d[0], c1d[1], c1d[2], c1d[3]);

        // Stage 3: Conv2 + GELU
        var conv2 = new Conv1d(weights.get("encoder.conv2.weight"), weights.get("encoder.conv2.bias"), 2, 1);
        Tensor c2 = conv2.forward(c1).gelu();
        float[] c2d = c2.data();
        System.out.printf("[Stage 3] Conv2+GELU: shape=%s first4=[%.8f, %.8f, %.8f, %.8f]%n",
                c2, c2d[0], c2d[1], c2d[2], c2d[3]);

        // Stage 4: Transpose + positional embedding
        Tensor x = c2.transpose();
        Tensor pe = weights.get("encoder.positional_embedding");
        x = x.add(pe);
        float[] xd = x.data();
        System.out.printf("[Stage 4] After pos_emb: shape=%s first4=[%.8e, %.8e, %.8e, %.8e]%n",
                x, xd[0], xd[1], xd[2], xd[3]);

        // Stage 5: Encoder block 0
        var enc = new WhisperEncoder(weights, dims);
        // Can't easily run just block 0, so run full encoder
        Tensor encOut = enc.forward(mel3d);
        float[] ed = encOut.data();
        System.out.printf("[Stage 6] Encoder output: shape=%s first8=[%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]%n",
                encOut, ed[0], ed[1], ed[2], ed[3], ed[4], ed[5], ed[6], ed[7]);

        // Stage 7: Decoder
        var dec = new WhisperDecoder(weights, dims);
        int[] prompt = {50257, 50258, 50359, 50363};
        System.out.printf("[Stage 7] Decoder prompt: %s%n", java.util.Arrays.toString(prompt));

        Tensor logits = dec.forward(prompt, encOut, null);
        float[] ld = logits.data();
        int nVocab = logits.dim(2);
        int lastPos = logits.dim(1) - 1;
        float[] lastLogits = new float[nVocab];
        System.arraycopy(ld, lastPos * nVocab, lastLogits, 0, nVocab);

        // Top 5
        int[] top5 = topK(lastLogits, 5);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 5; i++) {
            if (i > 0) sb.append(", ");
            sb.append("(").append(top5[i]).append(", ").append(String.format("%.3f", lastLogits[top5[i]])).append(")");
        }
        sb.append("]");
        System.out.printf("[Stage 7] Top 5 tokens: %s%n", sb);
        System.out.printf("[Stage 7] EOT score: %.3f%n", lastLogits[50256]);
    }

    static float mean(float[] a) { double s = 0; for (float v : a) s += v; return (float)(s / a.length); }

    static int[] topK(float[] a, int k) {
        int[] idx = new int[k];
        float[] vals = new float[k];
        java.util.Arrays.fill(vals, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < a.length; i++) {
            if (a[i] > vals[k-1]) {
                vals[k-1] = a[i]; idx[k-1] = i;
                for (int j = k-2; j >= 0; j--) {
                    if (vals[j+1] > vals[j]) {
                        float tv = vals[j]; vals[j] = vals[j+1]; vals[j+1] = tv;
                        int ti = idx[j]; idx[j] = idx[j+1]; idx[j+1] = ti;
                    }
                }
            }
        }
        return idx;
    }
}

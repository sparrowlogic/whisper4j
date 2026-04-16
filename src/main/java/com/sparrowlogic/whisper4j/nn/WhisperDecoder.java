package com.sparrowlogic.whisper4j.nn;

import com.sparrowlogic.whisper4j.model.ModelDimensions;
import com.sparrowlogic.whisper4j.model.WeightStore;
import com.sparrowlogic.whisper4j.tensor.Tensor;
import org.jspecify.annotations.Nullable;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.zip.Deflater;

/**
 * Whisper text decoder with temperature fallback, token suppression,
 * compression ratio check, and no-speech detection.
 * Ported from openai/whisper (cba3768) whisper/decoding.py.
 */
@SuppressWarnings({"checkstyle:ClassDataAbstractionCoupling", "checkstyle:ClassFanOutComplexity"})
public final class WhisperDecoder {

    private static final Logger LOG = Logger.getLogger(WhisperDecoder.class.getName());
    private static final ValueLayout.OfFloat FLOAT_LE =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final Tensor tokenEmbedding;
    private final Tensor positionalEmbedding;
    private final ResidualAttentionBlock[] blocks;
    private final LayerNorm ln;
    private final Tensor causalMask;

    public WhisperDecoder(final WeightStore w, final ModelDimensions dims) {
        String p = "decoder.";
        this.tokenEmbedding = w.get(p + "token_embedding.weight");
        this.positionalEmbedding = w.get(p + "positional_embedding");
        this.ln = new LayerNorm(w.get(p + "ln.weight"), w.get(p + "ln.bias"));
        int nTextCtx = dims.nTextCtx();

        float[] maskData = new float[nTextCtx * nTextCtx];
        for (int i = 0; i < nTextCtx; i++) {
            for (int j = i + 1; j < nTextCtx; j++) {
                maskData[i * nTextCtx + j] = Float.NEGATIVE_INFINITY;
            }
        }
        this.causalMask = Tensor.of(maskData, nTextCtx, nTextCtx);

        this.blocks = new ResidualAttentionBlock[dims.nTextLayer()];
        for (int i = 0; i < dims.nTextLayer(); i++) {
            String bp = p + "blocks." + i + ".";
            this.blocks[i] = buildDecoderBlock(w, bp, dims.nTextHead());
        }
    }

    /** Forward pass: tokens → logits. */
    public Tensor forward(final int[] tokens, final Tensor xa,
                          final @Nullable Map<String, Tensor> kvCache) {
        return this.forwardWithAttn(tokens, xa, kvCache, null);
    }

    /** Forward pass capturing cross-attention weights from specified layers. */
    @SuppressWarnings("checkstyle:ExecutableStatementCount")
    public Tensor forwardWithAttn(final int[] tokens, final Tensor xa,
                                  final @Nullable Map<String, Tensor> kvCache,
                                  final @Nullable List<Tensor> crossAttnOut) {
        int seqLen = tokens.length;
        int nState = this.tokenEmbedding.dim(1);
        int nCtx = this.positionalEmbedding.dim(0);
        int offset = kvCache != null ? Math.min(this.inferOffset(kvCache), nCtx - seqLen) : 0;

        Tensor x = Tensor.allocateNative(1, seqLen, nState);
        long hdBytes = (long) nState * Float.BYTES;
        for (int t = 0; t < seqLen; t++) {
            long dstOff = (long) t * nState * Float.BYTES;
            long teOff = (long) tokens[t] * nState * Float.BYTES;
            long peOff = (long) (offset + t) * nState * Float.BYTES;
            MemorySegment.copy(this.tokenEmbedding.segment(), teOff, x.segment(), dstOff, hdBytes);
            for (int d = 0; d < nState; d++) {
                long idx = dstOff + (long) d * Float.BYTES;
                x.segment().set(FLOAT_LE, idx,
                        x.segment().get(FLOAT_LE, idx)
                                + this.positionalEmbedding.segment().get(FLOAT_LE, peOff + (long) d * Float.BYTES));
            }
        }

        for (int i = 0; i < this.blocks.length; i++) {
            if (crossAttnOut != null) {
                Tensor[] result = this.blocks[i].forwardWithAttn(
                        x, xa, this.causalMask, kvCache, "decoder.blocks." + i);
                x = result[0];
                if (result[1] != null) {
                    crossAttnOut.add(result[1]);
                }
            } else {
                x = this.blocks[i].forward(x, xa, this.causalMask, kvCache, "decoder.blocks." + i);
            }
        }

        x = this.ln.forward(x);
        return x.matmulTransB(this.tokenEmbedding, 1.0f);
    }

    /**
     * Decode with temperature fallback matching whisper reference.
     * Retries with higher temperature if compression ratio &gt; 2.4 or avg logprob &lt; -1.0.
     */
    public DecodeResult decodeWithFallback(final Tensor encoderOutput, final int[] prompt,
                                           final int eotToken, final int maxTokens,
                                           final int[] suppressTokens, final int noSpeechToken) {
        return this.decodeWithFallback(encoderOutput, prompt, eotToken, maxTokens,
                suppressTokens, noSpeechToken, -1, 1);
    }

    /** Decode with temperature fallback, beam search at t=0, greedy sampling at t&gt;0. */
    @SuppressWarnings("checkstyle:ExecutableStatementCount")
    public DecodeResult decodeWithFallback(final Tensor encoderOutput, final int[] prompt,
                                           final int eotToken, final int maxTokens,
                                           final int[] suppressTokens, final int noSpeechToken,
                                           final int timestampBegin, final int beamSize) {
        float[] temperatures = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
        DecodeResult best = null;

        for (final float temp : temperatures) {
            DecodeResult result;
            if (beamSize > 1 && temp == 0.0f) {
                result = this.decodeBeam(encoderOutput, prompt, eotToken, maxTokens,
                        suppressTokens, noSpeechToken, timestampBegin, beamSize);
            } else {
                result = this.decode(encoderOutput, prompt, eotToken, maxTokens,
                        suppressTokens, noSpeechToken, temp, timestampBegin);
            }
            if (!needsFallback(result)) {
                return result;
            }
            if (best == null || result.avgLogprob > best.avgLogprob) {
                best = result;
            }
            if (temp == 0.0f && result.compressionRatio <= 2.4f) {
                break;
            }
            if (temp > 0.0f && result.avgLogprob < best.avgLogprob - 1.0f) {
                break;
            }
        }
        return best;
    }

    public DecodeResult decode(final Tensor encoderOutput, final int[] prompt,
                               final int eotToken, final int maxTokens,
                               final int[] suppressTokens, final int noSpeechToken,
                               final float temperature) {
        return this.decode(encoderOutput, prompt, eotToken, maxTokens, suppressTokens, noSpeechToken, temperature, -1);
    }

    @SuppressWarnings({"checkstyle:ExecutableStatementCount",
        "checkstyle:CyclomaticComplexity", "checkstyle:NPathComplexity",
        "checkstyle:ParameterNumber"})
    public DecodeResult decode(final Tensor encoderOutput, final int[] prompt,
                               final int eotToken, final int maxTokens,
                               final int[] suppressTokens, final int noSpeechToken,
                               final float temperature, final int timestampBegin) {
        Map<String, Tensor> kvCache = new HashMap<>();
        int[] generated = new int[maxTokens];
        int[] currentTokens = prompt;
        int genCount = 0;
        double sumLogprob = 0;
        float noSpeechProb = 0;
        int nVocab = this.tokenEmbedding.dim(0);

        for (int step = 0; step < maxTokens; step++) {
            Tensor logits = this.forward(currentTokens, encoderOutput, kvCache);
            int lastPos = logits.dim(1) - 1;
            float[] last = logits.getRow(lastPos);

            if (step == 0) {
                noSpeechProb = this.collectNoSpeechProb(logits, noSpeechToken, nVocab);
                this.suppressBlank(last, eotToken);
            }
            this.applySuppress(last, suppressTokens, nVocab);
            applyRepetitionPenalty(last, generated, genCount, nVocab);

            // Apply timestamp rules if decoding with timestamps
            if (timestampBegin > 0) {
                applyTimestampRules(last, generated, genCount, timestampBegin, eotToken, nVocab, step == 0);
            }

            int bestToken = temperature == 0.0f ? argmax(last) : sample(last, temperature);
            sumLogprob += logSoftmax(last, bestToken);

            if (bestToken == eotToken) {
                break;
            }
            generated[genCount++] = bestToken;
            currentTokens = new int[]{bestToken};
        }

        int[] result = new int[genCount];
        System.arraycopy(generated, 0, result, 0, genCount);
        float avgLp = genCount > 0 ? (float) (sumLogprob / genCount) : 0;
        return new DecodeResult(result, avgLp, compressionRatio(result), noSpeechProb, temperature);
    }

    /**
     * Beam search decode matching whisper reference (decoding.py BeamSearchDecoder).
     * Maintains beamSize separate KV caches. Each step: forward per beam → expand
     * by top-(beamSize+1) → rank globally → keep top beamSize → rearrange caches.
     * Cross-attention caches are shared (populated once from encoder output).
     */
    @SuppressWarnings({"checkstyle:ExecutableStatementCount", "checkstyle:CyclomaticComplexity",
        "checkstyle:NPathComplexity", "checkstyle:ParameterNumber", "checkstyle:NestedForDepth"})
    public DecodeResult decodeBeam(final Tensor encoderOutput, final int[] prompt,
                                    final int eotToken, final int maxTokens,
                                    final int[] suppressTokens, final int noSpeechToken,
                                    final int timestampBegin, final int beamSize) {
        int nVocab = this.tokenEmbedding.dim(0);
        float noSpeechProb = 0;

        // Each beam has its own KV cache and token sequence
        @SuppressWarnings("unchecked")
        Map<String, Tensor>[] caches = new HashMap[beamSize];
        int[][] beamTokens = new int[beamSize][maxTokens];
        int[] beamLengths = new int[beamSize];
        double[] beamLogprobs = new double[beamSize];
        for (int b = 0; b < beamSize; b++) {
            caches[b] = new HashMap<>();
        }

        // Finished sequences: tokens + sumLogprob
        List<int[]> finishedTokens = new ArrayList<>();
        List<Double> finishedLogprobs = new ArrayList<>();

        // Step 0: run prompt through beam 0, then copy cross-attention caches to all beams
        Tensor logits0 = this.forward(prompt, encoderOutput, caches[0]);
        noSpeechProb = this.collectNoSpeechProb(logits0, noSpeechToken, nVocab);

        // Copy cross-attention caches from beam 0 to all other beams
        for (int b = 1; b < beamSize; b++) {
            for (var entry : caches[0].entrySet()) {
                if (entry.getKey().contains("cross_attn")) {
                    caches[b].put(entry.getKey(), entry.getValue());
                }
            }
        }

        // Get initial top-k tokens from prompt logits
        float[] lastLogits = logits0.getRow(logits0.dim(1) - 1);
        this.suppressBlank(lastLogits, eotToken);
        this.applySuppress(lastLogits, suppressTokens, nVocab);
        float[] logprobs0 = computeLogSoftmax(lastLogits, nVocab);
        int[] topk = topkIndices(logprobs0, beamSize, nVocab);

        // Initialize beams with top-k tokens from step 0
        for (int b = 0; b < beamSize; b++) {
            beamTokens[b][0] = topk[b];
            beamLengths[b] = 1;
            beamLogprobs[b] = logprobs0[topk[b]];
            // Copy self-attention caches from beam 0 (all beams start from same prompt)
            for (var entry : caches[0].entrySet()) {
                if (!entry.getKey().contains("cross_attn")) {
                    caches[b].put(entry.getKey(), entry.getValue());
                }
            }
        }

        // Main beam search loop
        for (int step = 1; step < maxTokens; step++) {
            // Collect candidates: (score, beamIdx, token)
            List<double[]> candidates = new ArrayList<>();

            for (int b = 0; b < beamSize; b++) {
                if (beamTokens[b][beamLengths[b] - 1] == eotToken) {
                    continue; // beam already finished
                }
                int lastToken = beamTokens[b][beamLengths[b] - 1];
                Tensor logits = this.forward(new int[]{lastToken}, encoderOutput, caches[b]);
                float[] last = logits.getRow(logits.dim(1) - 1);

                this.applySuppress(last, suppressTokens, nVocab);
                applyRepetitionPenalty(last, beamTokens[b], beamLengths[b], nVocab);
                if (timestampBegin > 0) {
                    applyTimestampRules(last, beamTokens[b], beamLengths[b],
                            timestampBegin, eotToken, nVocab, false);
                }

                float[] lp = computeLogSoftmax(last, nVocab);
                int[] top = topkIndices(lp, beamSize + 1, nVocab);
                for (int k = 0; k < top.length; k++) {
                    candidates.add(new double[]{beamLogprobs[b] + lp[top[k]], b, top[k]});
                }
            }

            if (candidates.isEmpty()) {
                break; // all beams finished
            }

            // Sort candidates by score descending
            candidates.sort((a, b) -> Double.compare(b[0], a[0]));

            // Select top beamSize non-EOT candidates; EOT goes to finished
            int[][] newBeamTokens = new int[beamSize][maxTokens];
            int[] newBeamLengths = new int[beamSize];
            double[] newBeamLogprobs = new double[beamSize];
            int[] sourceBeams = new int[beamSize];
            int saved = 0;

            for (double[] cand : candidates) {
                int srcBeam = (int) cand[1];
                int token = (int) cand[2];
                double score = cand[0];

                if (token == eotToken) {
                    // Collect finished sequence
                    int[] finished = new int[beamLengths[srcBeam]];
                    System.arraycopy(beamTokens[srcBeam], 0, finished, 0, beamLengths[srcBeam]);
                    finishedTokens.add(finished);
                    finishedLogprobs.add(score);
                    if (finishedTokens.size() >= beamSize) {
                        break;
                    }
                    continue;
                }

                if (saved >= beamSize) {
                    continue;
                }

                System.arraycopy(beamTokens[srcBeam], 0, newBeamTokens[saved], 0, beamLengths[srcBeam]);
                newBeamTokens[saved][beamLengths[srcBeam]] = token;
                newBeamLengths[saved] = beamLengths[srcBeam] + 1;
                newBeamLogprobs[saved] = score;
                sourceBeams[saved] = srcBeam;
                saved++;
            }

            if (saved == 0 || finishedTokens.size() >= beamSize) {
                break;
            }

            // Rearrange KV caches: copy source beam's cache to new beam position
            @SuppressWarnings("unchecked")
            Map<String, Tensor>[] newCaches = new HashMap[beamSize];
            for (int b = 0; b < saved; b++) {
                if (sourceBeams[b] == b) {
                    newCaches[b] = caches[b]; // same beam, reuse cache
                } else {
                    newCaches[b] = new HashMap<>();
                    for (var entry : caches[sourceBeams[b]].entrySet()) {
                        newCaches[b].put(entry.getKey(), entry.getValue());
                    }
                }
            }
            // Fill remaining beams with copies of beam 0
            for (int b = saved; b < beamSize; b++) {
                newCaches[b] = newCaches[0];
                System.arraycopy(newBeamTokens[0], 0, newBeamTokens[b], 0, newBeamLengths[0]);
                newBeamLengths[b] = newBeamLengths[0];
                newBeamLogprobs[b] = newBeamLogprobs[0];
            }

            caches = newCaches;
            beamTokens = newBeamTokens;
            beamLengths = newBeamLengths;
            beamLogprobs = newBeamLogprobs;
        }

        // Add unfinished beams if not enough finished sequences
        if (finishedTokens.size() < beamSize) {
            for (int b = 0; b < beamSize && finishedTokens.size() < beamSize; b++) {
                int[] seq = new int[beamLengths[b]];
                System.arraycopy(beamTokens[b], 0, seq, 0, beamLengths[b]);
                finishedTokens.add(seq);
                finishedLogprobs.add(beamLogprobs[b]);
            }
        }

        // Select best by length-normalized logprob (matching MaximumLikelihoodRanker)
        int bestIdx = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < finishedTokens.size(); i++) {
            double score = finishedLogprobs.get(i) / Math.max(1, finishedTokens.get(i).length);
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        int[] bestTokens = finishedTokens.get(bestIdx);
        float avgLp = bestTokens.length > 0
                ? (float) (finishedLogprobs.get(bestIdx) / bestTokens.length) : 0;
        return new DecodeResult(bestTokens, avgLp, compressionRatio(bestTokens), noSpeechProb, 0.0f);
    }

    /** Compute log-softmax over logits array. */
    private static float[] computeLogSoftmax(final float[] logits, final int n) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            if (logits[i] > max) { max = logits[i]; }
        }
        double logSum = 0;
        for (int i = 0; i < n; i++) {
            logSum += Math.exp(logits[i] - max);
        }
        float logSumF = (float) (max + Math.log(logSum));
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = logits[i] - logSumF;
        }
        return out;
    }

    /** Return indices of top-k values in descending order. */
    private static int[] topkIndices(final float[] values, final int k, final int n) {
        int[] indices = new int[k];
        float[] topVals = new float[k];
        java.util.Arrays.fill(topVals, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < n; i++) {
            if (values[i] > topVals[k - 1]) {
                // Insert into sorted position
                int pos = k - 1;
                while (pos > 0 && values[i] > topVals[pos - 1]) { pos--; }
                System.arraycopy(topVals, pos, topVals, pos + 1, k - 1 - pos);
                System.arraycopy(indices, pos, indices, pos + 1, k - 1 - pos);
                topVals[pos] = values[i];
                indices[pos] = i;
            }
        }
        return indices;
    }

    // ---- Helpers ----

    private static boolean needsFallback(final DecodeResult r) {
        if (r.compressionRatio > 2.4f) {
            return !(r.noSpeechProb > 0.6f && r.avgLogprob < -1.0f);
        }
        return r.avgLogprob < -1.0f && !(r.noSpeechProb > 0.6f);
    }

    private float collectNoSpeechProb(final Tensor logits, final int noSpeechToken, final int nVocab) {
        if (noSpeechToken < 0 || noSpeechToken >= nVocab) {
            return 0;
        }
        float[] sotLogits = logits.getRow(0);
        return softmaxAt(sotLogits, noSpeechToken);
    }

    private void suppressBlank(final float[] logits, final int eotToken) {
        logits[eotToken] = Float.NEGATIVE_INFINITY;
        if (220 < logits.length) {
            logits[220] = Float.NEGATIVE_INFINITY;
        }
    }

    private void applySuppress(final float[] logits, final int[] tokens, final int nVocab) {
        if (tokens == null) {
            return;
        }
        for (final int t : tokens) {
            if (t >= 0 && t < nVocab) {
                logits[t] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    /**
     * No-repeat n-gram penalty: if the last (n-1) tokens match any previous (n-1)-gram,
     * suppress the token that would complete the repeated n-gram. Uses n=3 and n=4.
     */
    private static void applyRepetitionPenalty(final float[] logits, final int[] generated,
                                               final int genCount, final int nVocab) {
        applyNgramBlock(logits, generated, genCount, 3, nVocab);
        applyNgramBlock(logits, generated, genCount, 4, nVocab);
    }

    /**
     * Apply timestamp token rules matching whisper reference (decoding.py ApplyTimestampRules).
     * Enforces: timestamps in pairs, non-decreasing, first token must be timestamp.
     */
    @SuppressWarnings({"checkstyle:CyclomaticComplexity", "checkstyle:NPathComplexity",
        "checkstyle:ReturnCount"})
    private static void applyTimestampRules(final float[] logits, final int[] generated,
                                            final int genCount, final int tsBegin,
                                            final int eotToken, final int nVocab,
                                            final boolean isFirstStep) {
        // Always suppress <|notimestamps|> when decoding with timestamps
        if (tsBegin > 0 && tsBegin - 1 >= 0 && tsBegin - 1 < nVocab) {
            logits[tsBegin - 1] = Float.NEGATIVE_INFINITY;
        }

        // At first step, force timestamp token with max_initial_timestamp=1.0s (index 50)
        if (isFirstStep) {
            for (int i = 0; i < tsBegin; i++) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
            int maxInitialIdx = 50; // 1.0s / 0.02s precision
            int lastAllowed = tsBegin + maxInitialIdx;
            for (int i = lastAllowed + 1; i < nVocab; i++) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
            return;
        }

        if (genCount == 0) {
            return;
        }

        boolean lastWasTs = generated[genCount - 1] >= tsBegin;
        boolean penultimateWasTs = genCount < 2 || generated[genCount - 2] >= tsBegin;

        if (lastWasTs) {
            if (penultimateWasTs) {
                // Two consecutive timestamps — force non-timestamp next
                for (int i = tsBegin; i < nVocab; i++) {
                    logits[i] = Float.NEGATIVE_INFINITY;
                }
            } else {
                // Single timestamp — force either timestamp or EOT (no text)
                for (int i = 0; i < eotToken; i++) {
                    logits[i] = Float.NEGATIVE_INFINITY;
                }
            }
        }

        // Timestamps must not decrease
        int lastTs = -1;
        for (int i = genCount - 1; i >= 0; i--) {
            if (generated[i] >= tsBegin) {
                lastTs = generated[i];
                break;
            }
        }
        if (lastTs >= 0) {
            int blockUpTo = lastWasTs && !penultimateWasTs ? lastTs + 1 : lastTs;
            for (int i = tsBegin; i < blockUpTo && i < nVocab; i++) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    private static void applyNgramBlock(final float[] logits, final int[] generated,
                                        final int genCount, final int n, final int nVocab) {
        if (genCount < n - 1) {
            return;
        }
        // Current suffix: last (n-1) tokens
        for (int i = 0; i <= genCount - n; i++) {
            boolean match = true;
            for (int j = 0; j < n - 1; j++) {
                if (generated[i + j] != generated[genCount - (n - 1) + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                int blocked = generated[i + n - 1];
                if (blocked >= 0 && blocked < nVocab) {
                    logits[blocked] = Float.NEGATIVE_INFINITY;
                }
            }
        }
    }

    private static int argmax(final float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }

    private static int sample(final float[] logits, final float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float v : logits) {
            if (v > max) {
                max = v;
            }
        }
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            double p = Math.exp((logits[i] - max) / temperature);
            logits[i] = (float) p;
            sum += p;
        }
        double r = Math.random() * sum;
        double cum = 0;
        for (int i = 0; i < logits.length; i++) {
            cum += logits[i];
            if (cum >= r) {
                return i;
            }
        }
        return logits.length - 1;
    }

    private static float softmaxAt(final float[] logits, final int index) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float v : logits) {
            if (v > max) {
                max = v;
            }
        }
        double sum = 0;
        for (final float v : logits) {
            sum += Math.exp(v - max);
        }
        return (float) (Math.exp(logits[index] - max) / sum);
    }

    private static double logSoftmax(final float[] logits, final int index) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float v : logits) {
            if (v > max) {
                max = v;
            }
        }
        double logSum = 0;
        for (final float v : logits) {
            logSum += Math.exp(v - max);
        }
        return (logits[index] - max) - Math.log(logSum);
    }

    static float compressionRatio(final int[] tokens) {
        if (tokens.length == 0) {
            return 0;
        }
        byte[] bytes = new byte[tokens.length * 2];
        for (int i = 0; i < tokens.length; i++) {
            bytes[i * 2] = (byte) (tokens[i] & 0xFF);
            bytes[i * 2 + 1] = (byte) ((tokens[i] >> 8) & 0xFF);
        }
        Deflater deflater = new Deflater();
        deflater.setInput(bytes);
        deflater.finish();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[1024];
        while (!deflater.finished()) {
            bos.write(buf, 0, deflater.deflate(buf));
        }
        deflater.end();
        return (float) bytes.length / Math.max(1, bos.size());
    }

    private int inferOffset(final Map<String, Tensor> kvCache) {
        for (final var entry : kvCache.entrySet()) {
            if (entry.getKey().contains("blocks.0.attn.kr")) {
                return entry.getValue().dim(1);
            }
        }
        return 0;
    }

    private static ResidualAttentionBlock buildDecoderBlock(final WeightStore w, final String bp,
                                                            final int nHead) {
        return new ResidualAttentionBlock(
                WhisperEncoder.buildMHA(w, bp + "attn.", nHead),
                new LayerNorm(w.get(bp + "attn_ln.weight"), w.get(bp + "attn_ln.bias")),
                WhisperEncoder.buildMHA(w, bp + "cross_attn.", nHead),
                new LayerNorm(w.get(bp + "cross_attn_ln.weight"), w.get(bp + "cross_attn_ln.bias")),
                new Linear(w.get(bp + "mlp.0.weight"), w.get(bp + "mlp.0.bias")),
                new Linear(w.get(bp + "mlp.2.weight"), w.get(bp + "mlp.2.bias")),
                new LayerNorm(w.get(bp + "mlp_ln.weight"), w.get(bp + "mlp_ln.bias"))
        );
    }

    /** Decode result with metadata for temperature fallback decisions. */
    public record DecodeResult(int[] tokens, float avgLogprob,
                               float compressionRatio, float noSpeechProb, float temperature) { }
}

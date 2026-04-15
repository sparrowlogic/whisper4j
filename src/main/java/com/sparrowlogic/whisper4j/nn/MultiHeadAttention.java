package com.sparrowlogic.whisper4j.nn;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.Map;

/**
 * Multi-head attention with BLAS-accelerated matmul.
 * Caches K/V in rearranged-heads format to avoid full re-rearrange every step.
 * Uses a fused attention kernel for single-token (M=1) decoding that combines
 * QK^T, softmax, and AV into one pass per head with zero intermediate allocations.
 */
@SuppressWarnings({
    "checkstyle:CyclomaticComplexity",
    "checkstyle:ExecutableStatementCount",
    "checkstyle:MultipleStringLiterals",
    "checkstyle:NestedForDepth"
})
public final class MultiHeadAttention {

    private final int nHead;
    private final Linear query;
    private final Linear key;
    private final Linear value;
    private final Linear out;

    public MultiHeadAttention(final int nHead, final Linear query, final Linear key,
                              final Linear value, final Linear out) {
        this.nHead = nHead;
        this.query = query;
        this.key = key;
        this.value = value;
        this.out = out;
    }

    @SuppressWarnings("checkstyle:NestedIfDepth")
    public Tensor[] forward(final Tensor x, final Tensor xa, final Tensor mask,
                            final Map<String, Tensor> kvCache, final String cachePrefix) {
        Tensor q = this.query.forward(x);
        int batch = q.dim(0);
        int qLen = q.dim(1);
        int state = q.dim(2);
        int headDim = state / this.nHead;
        float scale = (float) (1.0 / Math.sqrt(headDim));

        String krKey = cachePrefix + ".kr";
        String vrKey = cachePrefix + ".vr";

        Tensor kr;
        Tensor vr;

        if (kvCache != null && xa != null && kvCache.containsKey(krKey)) {
            kr = kvCache.get(krKey);
            vr = kvCache.get(vrKey);
        } else if (kvCache != null && xa == null && kvCache.containsKey(krKey)) {
            Tensor newK = this.key.forward(x);
            Tensor newV = this.value.forward(x);
            Tensor newKr = newK.rearrangeToHeads(batch, qLen, this.nHead, headDim);
            Tensor newVr = newV.rearrangeToHeads(batch, qLen, this.nHead, headDim);
            kr = appendHeadsCache(kvCache, krKey, newKr);
            vr = appendHeadsCache(kvCache, vrKey, newVr);
        } else {
            Tensor source = xa != null ? xa : x;
            Tensor k = this.key.forward(source);
            Tensor v = this.value.forward(source);
            kr = k.rearrangeToHeads(batch, k.dim(1), this.nHead, headDim);
            vr = v.rearrangeToHeads(batch, v.dim(1), this.nHead, headDim);
            if (kvCache != null) {
                kvCache.put(krKey, kr);
                kvCache.put(vrKey, vr);
            }
        }

        int kvLen = kr.dim(1);

        // Fused single-token attention: QK^T + softmax + AV in one pass per head
        if (qLen == 1) {
            return this.fusedSingleTokenAttention(q, kr, vr, batch, kvLen, headDim, scale);
        }

        Tensor qr = q.rearrangeToHeads(batch, qLen, this.nHead, headDim);
        Tensor attnWeights = qr.matmulTransB(kr, scale);

        if (mask != null && qLen > 1) {
            attnWeights.addMaskInPlace(mask, batch * this.nHead, qLen, kvLen);
        }

        attnWeights = attnWeights.softmax();
        Tensor attnOut = attnWeights.matmul(vr);
        Tensor merged = attnOut.rearrangeFromHeads(batch, qLen, this.nHead, headDim);

        return new Tensor[]{this.out.forward(merged), attnWeights};
    }

    /**
     * Fused attention for single-token query (M=1 decoder step).
     * Combines QK^T, softmax, and AV into one pass per head with zero
     * intermediate tensor allocations. Eliminates: rearrangeToHeads(q),
     * matmulTransB, softmax tensor, matmul, rearrangeFromHeads.
     *
     * <p>q layout: (batch, 1, nHead*headDim) — head data is interleaved.
     * kr/vr layout: (batch*nHead, kvLen, headDim) — contiguous per head.
     * Output: (batch, 1, nHead*headDim) written directly for out projection.
     */
    @SuppressWarnings({"checkstyle:ParameterNumber", "checkstyle:LocalVariableName",
        "checkstyle:NestedForDepth"})
    private Tensor[] fusedSingleTokenAttention(final Tensor q, final Tensor kr, final Tensor vr,
                                                final int batch, final int kvLen,
                                                final int headDim, final float scale) {
        int state = this.nHead * headDim;
        Tensor merged = Tensor.allocateNative(batch, 1, state);
        float[] scores = new float[kvLen];

        for (int b = 0; b < batch; b++) {
            long qBase = (long) b * state * Float.BYTES;

            for (int h = 0; h < this.nHead; h++) {
                long qOff = qBase + (long) h * headDim * Float.BYTES;
                long kvBase = (long) (b * this.nHead + h) * kvLen * headDim * Float.BYTES;
                long outOff = qBase + (long) h * headDim * Float.BYTES;

                Tensor.fusedAttentionHead(q.segment(), qOff, kr.segment(), kvBase,
                        vr.segment(), merged.segment(), outOff,
                        kvLen, headDim, scale, scores);
            }
        }

        return new Tensor[]{this.out.forward(merged), null};
    }

    /**
     * Append new rearranged-heads data to existing cache.
     * Matches Python whisper reference: torch.cat([cache, output], dim=1).
     * Cache layout: (batch*nHead, seqLen, headDim) — contiguous per head.
     * Allocates a new result tensor each step (same as torch.cat).
     */
    private static Tensor appendHeadsCache(final Map<String, Tensor> cache, final String key,
                                           final Tensor newVal) {
        Tensor existing = cache.get(key);
        if (existing == null) {
            cache.put(key, newVal);
            return newVal;
        }
        int batchHeads = existing.dim(0);
        int existLen = existing.dim(1);
        int newLen = newVal.dim(1);
        int headDim = existing.dim(2);
        int totalLen = existLen + newLen;
        long hdBytes = (long) headDim * Float.BYTES;

        Tensor result = Tensor.allocateNative(batchHeads, totalLen, headDim);
        for (int h = 0; h < batchHeads; h++) {
            long dstOff = (long) h * totalLen * hdBytes;
            MemorySegment.copy(existing.segment(), (long) h * existLen * hdBytes,
                    result.segment(), dstOff, existLen * hdBytes);
            MemorySegment.copy(newVal.segment(), (long) h * newLen * hdBytes,
                    result.segment(), dstOff + existLen * hdBytes, newLen * hdBytes);
        }
        cache.put(key, result);
        return result;
    }
}

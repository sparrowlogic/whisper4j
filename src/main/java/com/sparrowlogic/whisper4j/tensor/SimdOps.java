package com.sparrowlogic.whisper4j.tensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.logging.Logger;

/**
 * SIMD-accelerated element-wise operations with automatic fallback.
 * Uses the Vector API (jdk.incubator.vector) when available (Java 26+ with --enable-preview),
 * otherwise falls back to scalar loops. Detection happens once at class load time.
 */
final class SimdOps {

    private static final Logger LOG = Logger.getLogger(SimdOps.class.getName());
    static final ValueLayout.OfFloat FLOAT_LE =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static final SimdOps INSTANCE;
    private static final boolean VECTOR_API_AVAILABLE;

    // Vector API handles — null when not available
    private final VectorOpsDelegate delegate;

    static {
        VectorOpsDelegate d = null;
        boolean avail = false;
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            d = (VectorOpsDelegate) Class
                    .forName("com.sparrowlogic.whisper4j.tensor.VectorSimdOps")
                    .getDeclaredConstructor()
                    .newInstance();
            avail = true;
            LOG.info("Vector API available — SIMD acceleration enabled");
        } catch (final Throwable t) {
            LOG.info("Vector API not available — using scalar fallback: " + t.getMessage());
        }
        VECTOR_API_AVAILABLE = avail;
        INSTANCE = new SimdOps(d);
    }

    private SimdOps(final VectorOpsDelegate delegate) {
        this.delegate = delegate;
    }

    static SimdOps get() {
        return INSTANCE;
    }

    static boolean isVectorApiAvailable() {
        return VECTOR_API_AVAILABLE;
    }

    // ---- Dot product ----

    float dot(final MemorySegment a, final long aByteOff,
              final MemorySegment b, final long bByteOff, final int len) {
        if (this.delegate != null) {
            return this.delegate.dot(a, aByteOff, b, bByteOff, len);
        }
        return scalarDot(a, aByteOff, b, bByteOff, len);
    }

    // ---- Binary add (same-size) ----

    void add(final MemorySegment a, final MemorySegment b,
             final MemorySegment out, final int len) {
        if (this.delegate != null) {
            this.delegate.add(a, b, out, len);
            return;
        }
        scalarAdd(a, 0, b, 0, out, 0, len);
    }

    // ---- Binary add with offsets ----

    void addOffset(final MemorySegment a, final long aOff,
                   final MemorySegment b, final long bOff,
                   final MemorySegment out, final long oOff, final int len) {
        if (this.delegate != null) {
            this.delegate.addOffset(a, aOff, b, bOff, out, oOff, len);
            return;
        }
        scalarAdd(a, aOff, b, bOff, out, oOff, len);
    }

    // ---- Scale (multiply all by scalar) ----

    void scale(final MemorySegment src, final MemorySegment dst,
               final int size, final float s) {
        if (this.delegate != null) {
            this.delegate.scale(src, dst, size, s);
            return;
        }
        for (int i = 0; i < size; i++) {
            long byteI = (long) i * Float.BYTES;
            dst.set(FLOAT_LE, byteI, src.get(FLOAT_LE, byteI) * s);
        }
    }

    // ---- Scale in-place ----

    void scaleInPlace(final MemorySegment seg, final int size, final float s) {
        if (this.delegate != null) {
            this.delegate.scaleInPlace(seg, size, s);
            return;
        }
        for (int i = 0; i < size; i++) {
            long byteI = (long) i * Float.BYTES;
            seg.set(FLOAT_LE, byteI, seg.get(FLOAT_LE, byteI) * s);
        }
    }

    // ---- M=1 matrix-vector: out += x[k] * B[k, :] ----

    @SuppressWarnings("checkstyle:ParameterNumber")
    void matmulVecAccumulate(final MemorySegment xSeg, final long xOff, final float xk,
                             final MemorySegment bSeg, final long bRowOff,
                             final MemorySegment outSeg, final long cOff, final int n) {
        if (this.delegate != null) {
            this.delegate.matmulVecAccumulate(xSeg, xOff, xk, bSeg, bRowOff, outSeg, cOff, n);
            return;
        }
        float[] bRow = new float[n];
        float[] out = new float[n];
        MemorySegment.copy(bSeg, FLOAT_LE, bRowOff, bRow, 0, n);
        MemorySegment.copy(outSeg, FLOAT_LE, cOff, out, 0, n);
        for (int j = 0; j < n; j++) {
            out[j] += xk * bRow[j];
        }
        MemorySegment.copy(out, 0, outSeg, FLOAT_LE, cOff, n);
    }

    // ---- Fused single-token attention per head ----

    @SuppressWarnings({"checkstyle:ParameterNumber", "checkstyle:CyclomaticComplexity"})
    void fusedAttentionHead(final MemorySegment qSeg, final long qOff,
                            final MemorySegment kSeg, final long kvBase,
                            final MemorySegment vSeg,
                            final MemorySegment mSeg, final long outOff,
                            final int kvLen, final int headDim, final float scale,
                            final float[] scores) {
        if (this.delegate != null) {
            this.delegate.fusedAttentionHead(qSeg, qOff, kSeg, kvBase, vSeg, mSeg, outOff,
                    kvLen, headDim, scale, scores);
            return;
        }
        scalarFusedAttentionHead(qSeg, qOff, kSeg, kvBase, vSeg, mSeg, outOff,
                kvLen, headDim, scale, scores);
    }

    // ---- Scalar implementations ----

    static float scalarDot(final MemorySegment a, final long aByteOff,
                           final MemorySegment b, final long bByteOff, final int len) {
        float[] aa = new float[len];
        float[] bb = new float[len];
        MemorySegment.copy(a, FLOAT_LE, aByteOff, aa, 0, len);
        MemorySegment.copy(b, FLOAT_LE, bByteOff, bb, 0, len);
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += aa[i] * bb[i];
        }
        return sum;
    }

    private static void scalarAdd(final MemorySegment a, final long aOff,
                                  final MemorySegment b, final long bOff,
                                  final MemorySegment out, final long oOff, final int len) {
        float[] aa = new float[len];
        float[] bb = new float[len];
        MemorySegment.copy(a, FLOAT_LE, aOff, aa, 0, len);
        MemorySegment.copy(b, FLOAT_LE, bOff, bb, 0, len);
        for (int i = 0; i < len; i++) {
            aa[i] += bb[i];
        }
        MemorySegment.copy(aa, 0, out, FLOAT_LE, oOff, len);
    }

    @SuppressWarnings({"checkstyle:ParameterNumber", "checkstyle:CyclomaticComplexity"})
    static void scalarFusedAttentionHead(final MemorySegment qSeg, final long qOff,
                                         final MemorySegment kSeg, final long kvBase,
                                         final MemorySegment vSeg,
                                         final MemorySegment mSeg, final long outOff,
                                         final int kvLen, final int headDim, final float scale,
                                         final float[] scores) {
        float[] q = new float[headDim];
        float[] kRow = new float[headDim];
        MemorySegment.copy(qSeg, FLOAT_LE, qOff, q, 0, headDim);

        // QK^T
        float maxScore = Float.NEGATIVE_INFINITY;
        long hdBytes = (long) headDim * Float.BYTES;
        for (int j = 0; j < kvLen; j++) {
            MemorySegment.copy(kSeg, FLOAT_LE, kvBase + (long) j * hdBytes, kRow, 0, headDim);
            float dot = 0;
            for (int d = 0; d < headDim; d++) {
                dot += q[d] * kRow[d];
            }
            float s = dot * scale;
            scores[j] = s;
            if (s > maxScore) { maxScore = s; }
        }

        // Softmax
        float sum = 0;
        for (int j = 0; j < kvLen; j++) {
            scores[j] = (float) Math.exp(scores[j] - maxScore);
            sum += scores[j];
        }
        float invSum = 1.0f / sum;
        for (int j = 0; j < kvLen; j++) { scores[j] *= invSum; }

        // AV accumulate
        float[] outArr = new float[headDim];
        float[] vRow = new float[headDim];
        for (int j = 0; j < kvLen; j++) {
            float sj = scores[j];
            MemorySegment.copy(vSeg, FLOAT_LE, kvBase + (long) j * hdBytes, vRow, 0, headDim);
            for (int d = 0; d < headDim; d++) {
                outArr[d] += sj * vRow[d];
            }
        }
        MemorySegment.copy(outArr, 0, mSeg, FLOAT_LE, outOff, headDim);
    }

    private SimdOps() {
        this.delegate = null;
    }

    /** Interface for the Vector API delegate — loaded reflectively. */
    interface VectorOpsDelegate {
        float dot(MemorySegment a, long aByteOff, MemorySegment b, long bByteOff, int len);
        void add(MemorySegment a, MemorySegment b, MemorySegment out, int len);
        void addOffset(MemorySegment a, long aOff, MemorySegment b, long bOff,
                       MemorySegment out, long oOff, int len);
        void scale(MemorySegment src, MemorySegment dst, int size, float s);
        void scaleInPlace(MemorySegment seg, int size, float s);
        @SuppressWarnings("checkstyle:ParameterNumber")
        void matmulVecAccumulate(MemorySegment xSeg, long xOff, float xk,
                                 MemorySegment bSeg, long bRowOff,
                                 MemorySegment outSeg, long cOff, int n);
        @SuppressWarnings("checkstyle:ParameterNumber")
        void fusedAttentionHead(MemorySegment qSeg, long qOff,
                                MemorySegment kSeg, long kvBase,
                                MemorySegment vSeg,
                                MemorySegment mSeg, long outOff,
                                int kvLen, int headDim, float scale, float[] scores);
    }
}

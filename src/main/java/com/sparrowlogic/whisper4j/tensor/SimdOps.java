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
        for (int j = 0; j < n; j++) {
            long outIdx = cOff + (long) j * Float.BYTES;
            outSeg.set(FLOAT_LE, outIdx,
                    outSeg.get(FLOAT_LE, outIdx)
                            + xk * bSeg.get(FLOAT_LE, bRowOff + (long) j * Float.BYTES));
        }
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
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += a.getAtIndex(FLOAT_LE, aByteOff / Float.BYTES + i)
                    * b.getAtIndex(FLOAT_LE, bByteOff / Float.BYTES + i);
        }
        return sum;
    }

    private static void scalarAdd(final MemorySegment a, final long aOff,
                                  final MemorySegment b, final long bOff,
                                  final MemorySegment out, final long oOff, final int len) {
        for (int i = 0; i < len; i++) {
            long byteI = (long) i * Float.BYTES;
            out.set(FLOAT_LE, oOff + byteI,
                    a.get(FLOAT_LE, aOff + byteI) + b.get(FLOAT_LE, bOff + byteI));
        }
    }

    @SuppressWarnings({"checkstyle:ParameterNumber", "checkstyle:CyclomaticComplexity"})
    static void scalarFusedAttentionHead(final MemorySegment qSeg, final long qOff,
                                         final MemorySegment kSeg, final long kvBase,
                                         final MemorySegment vSeg,
                                         final MemorySegment mSeg, final long outOff,
                                         final int kvLen, final int headDim, final float scale,
                                         final float[] scores) {
        // QK^T
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < kvLen; j++) {
            long kOff = kvBase + (long) j * headDim * Float.BYTES;
            float dot = 0;
            for (int d = 0; d < headDim; d++) {
                dot += qSeg.get(FLOAT_LE, qOff + (long) d * Float.BYTES)
                        * kSeg.get(FLOAT_LE, kOff + (long) d * Float.BYTES);
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
        for (int d = 0; d < headDim; d++) {
            mSeg.set(FLOAT_LE, outOff + (long) d * Float.BYTES, 0f);
        }
        for (int j = 0; j < kvLen; j++) {
            float sj = scores[j];
            long vOff = kvBase + (long) j * headDim * Float.BYTES;
            for (int d = 0; d < headDim; d++) {
                long idx = outOff + (long) d * Float.BYTES;
                mSeg.set(FLOAT_LE, idx,
                        mSeg.get(FLOAT_LE, idx)
                                + sj * vSeg.get(FLOAT_LE, vOff + (long) d * Float.BYTES));
            }
        }
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

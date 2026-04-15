package com.sparrowlogic.whisper4j.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

/**
 * Vector API (JEP 529) implementation of SIMD operations.
 * This class is loaded reflectively by {@link SimdOps} — it is never referenced
 * directly from code that must run without the Vector API.
 */
@SuppressWarnings({"checkstyle:CyclomaticComplexity", "checkstyle:NestedForDepth"})
final class VectorSimdOps implements SimdOps.VectorOpsDelegate {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    @Override
    public float dot(final MemorySegment a, final long aByteOff,
                     final MemorySegment b, final long bByteOff, final int len) {
        int i = 0;
        int bound = SPECIES.loopBound(len);
        var acc = FloatVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length()) {
            long byteI = (long) i * Float.BYTES;
            var va = FloatVector.fromMemorySegment(SPECIES, a,
                    aByteOff + byteI, ByteOrder.LITTLE_ENDIAN);
            var vb = FloatVector.fromMemorySegment(SPECIES, b,
                    bByteOff + byteI, ByteOrder.LITTLE_ENDIAN);
            acc = acc.add(va.mul(vb));
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += a.getAtIndex(SimdOps.FLOAT_LE, aByteOff / Float.BYTES + i)
                    * b.getAtIndex(SimdOps.FLOAT_LE, bByteOff / Float.BYTES + i);
        }
        return sum;
    }

    @Override
    public void add(final MemorySegment a, final MemorySegment b,
                    final MemorySegment out, final int len) {
        int i = 0;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            long byteI = (long) i * Float.BYTES;
            var va = FloatVector.fromMemorySegment(SPECIES, a, byteI, ByteOrder.LITTLE_ENDIAN);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, byteI, ByteOrder.LITTLE_ENDIAN);
            va.add(vb).intoMemorySegment(out, byteI, ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < len; i++) {
            long byteI = (long) i * Float.BYTES;
            out.set(SimdOps.FLOAT_LE, byteI,
                    a.get(SimdOps.FLOAT_LE, byteI) + b.get(SimdOps.FLOAT_LE, byteI));
        }
    }

    @Override
    public void addOffset(final MemorySegment a, final long aOff,
                          final MemorySegment b, final long bOff,
                          final MemorySegment out, final long oOff, final int len) {
        int i = 0;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            long byteI = (long) i * Float.BYTES;
            var va = FloatVector.fromMemorySegment(SPECIES, a, aOff + byteI, ByteOrder.LITTLE_ENDIAN);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, bOff + byteI, ByteOrder.LITTLE_ENDIAN);
            va.add(vb).intoMemorySegment(out, oOff + byteI, ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < len; i++) {
            long byteI = (long) i * Float.BYTES;
            out.set(SimdOps.FLOAT_LE, oOff + byteI,
                    a.get(SimdOps.FLOAT_LE, aOff + byteI) + b.get(SimdOps.FLOAT_LE, bOff + byteI));
        }
    }

    @Override
    public void scale(final MemorySegment src, final MemorySegment dst,
                      final int size, final float s) {
        int i = 0;
        int bound = SPECIES.loopBound(size);
        var vs = FloatVector.broadcast(SPECIES, s);
        for (; i < bound; i += SPECIES.length()) {
            long byteI = (long) i * Float.BYTES;
            FloatVector.fromMemorySegment(SPECIES, src, byteI, ByteOrder.LITTLE_ENDIAN)
                    .mul(vs)
                    .intoMemorySegment(dst, byteI, ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < size; i++) {
            long byteI = (long) i * Float.BYTES;
            dst.set(SimdOps.FLOAT_LE, byteI, src.get(SimdOps.FLOAT_LE, byteI) * s);
        }
    }

    @Override
    public void scaleInPlace(final MemorySegment seg, final int size, final float s) {
        int i = 0;
        int bound = SPECIES.loopBound(size);
        var vs = FloatVector.broadcast(SPECIES, s);
        for (; i < bound; i += SPECIES.length()) {
            long byteI = (long) i * Float.BYTES;
            FloatVector.fromMemorySegment(SPECIES, seg, byteI, ByteOrder.LITTLE_ENDIAN)
                    .mul(vs).intoMemorySegment(seg, byteI, ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < size; i++) {
            long byteI = (long) i * Float.BYTES;
            seg.set(SimdOps.FLOAT_LE, byteI, seg.get(SimdOps.FLOAT_LE, byteI) * s);
        }
    }

    @Override
    @SuppressWarnings("checkstyle:ParameterNumber")
    public void matmulVecAccumulate(final MemorySegment xSeg, final long xOff, final float xk,
                                    final MemorySegment bSeg, final long bRowOff,
                                    final MemorySegment outSeg, final long cOff, final int n) {
        var vx = FloatVector.broadcast(SPECIES, xk);
        int j = 0;
        int bound = SPECIES.loopBound(n);
        for (; j < bound; j += SPECIES.length()) {
            long outIdx = cOff + (long) j * Float.BYTES;
            long bIdx = bRowOff + (long) j * Float.BYTES;
            var vc = FloatVector.fromMemorySegment(SPECIES, outSeg, outIdx, ByteOrder.LITTLE_ENDIAN);
            var vb = FloatVector.fromMemorySegment(SPECIES, bSeg, bIdx, ByteOrder.LITTLE_ENDIAN);
            vc.add(vx.mul(vb)).intoMemorySegment(outSeg, outIdx, ByteOrder.LITTLE_ENDIAN);
        }
        for (; j < n; j++) {
            long outIdx = cOff + (long) j * Float.BYTES;
            outSeg.set(SimdOps.FLOAT_LE, outIdx,
                    outSeg.get(SimdOps.FLOAT_LE, outIdx)
                            + xk * bSeg.get(SimdOps.FLOAT_LE, bRowOff + (long) j * Float.BYTES));
        }
    }

    @Override
    @SuppressWarnings({"checkstyle:ParameterNumber", "checkstyle:LocalVariableName"})
    public void fusedAttentionHead(final MemorySegment qSeg, final long qOff,
                                   final MemorySegment kSeg, final long kvBase,
                                   final MemorySegment vSeg,
                                   final MemorySegment mSeg, final long outOff,
                                   final int kvLen, final int headDim, final float scale,
                                   final float[] scores) {
        int simdBound = SPECIES.loopBound(headDim);

        // QK^T with SIMD dot
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < kvLen; j++) {
            long kOff = kvBase + (long) j * headDim * Float.BYTES;
            var acc = FloatVector.zero(SPECIES);
            int d = 0;
            for (; d < simdBound; d += SPECIES.length()) {
                long byteD = (long) d * Float.BYTES;
                var vq = FloatVector.fromMemorySegment(SPECIES, qSeg,
                        qOff + byteD, ByteOrder.LITTLE_ENDIAN);
                var vk = FloatVector.fromMemorySegment(SPECIES, kSeg,
                        kOff + byteD, ByteOrder.LITTLE_ENDIAN);
                acc = acc.add(vq.mul(vk));
            }
            float dot = acc.reduceLanes(VectorOperators.ADD);
            for (; d < headDim; d++) {
                dot += qSeg.get(SimdOps.FLOAT_LE, qOff + (long) d * Float.BYTES)
                        * kSeg.get(SimdOps.FLOAT_LE, kOff + (long) d * Float.BYTES);
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

        // AV accumulate with SIMD
        for (int d = 0; d < headDim; d++) {
            mSeg.set(SimdOps.FLOAT_LE, outOff + (long) d * Float.BYTES, 0f);
        }
        for (int j = 0; j < kvLen; j++) {
            float sj = scores[j];
            long vOff = kvBase + (long) j * headDim * Float.BYTES;
            var vs = FloatVector.broadcast(SPECIES, sj);
            int d = 0;
            for (; d < simdBound; d += SPECIES.length()) {
                long byteD = (long) d * Float.BYTES;
                var vo = FloatVector.fromMemorySegment(SPECIES, mSeg,
                        outOff + byteD, ByteOrder.LITTLE_ENDIAN);
                var vv = FloatVector.fromMemorySegment(SPECIES, vSeg,
                        vOff + byteD, ByteOrder.LITTLE_ENDIAN);
                vo.add(vs.mul(vv)).intoMemorySegment(mSeg,
                        outOff + byteD, ByteOrder.LITTLE_ENDIAN);
            }
            for (; d < headDim; d++) {
                long idx = outOff + (long) d * Float.BYTES;
                mSeg.set(SimdOps.FLOAT_LE, idx,
                        mSeg.get(SimdOps.FLOAT_LE, idx)
                                + sj * vSeg.get(SimdOps.FLOAT_LE, vOff + (long) d * Float.BYTES));
            }
        }
    }
}

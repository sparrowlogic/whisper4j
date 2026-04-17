package com.sparrowlogic.whisper4j.tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * N-dimensional tensor backed by off-heap {@link MemorySegment} in row-major order.
 * Uses the Foreign Memory API (Panama) to avoid GC heap pressure for model weights.
 * All heavy math is SIMD-accelerated via {@link SimdOps} when the Vector API is available.
 */
@SuppressWarnings({"checkstyle:MethodCount", "checkstyle:ClassDataAbstractionCoupling",
        "checkstyle:ClassFanOutComplexity"})
public final class Tensor {

    private static final ValueLayout.OfFloat FLOAT_LE = SimdOps.FLOAT_LE;
    private static final float[] GELU_TABLE = buildGeluTable();
    private static final ThreadLocal<Arena> SCOPED_ARENA = new ThreadLocal<>();

    private final MemorySegment segment;
    private final int[] shape;
    private final int[] strides;
    private final int size;

    private Tensor(final MemorySegment segment, final int[] shape) {
        this.segment = segment;
        this.shape = shape;
        this.strides = computeStrides(shape);
        this.size = elementCount(shape);
        long expectedBytes = (long) this.size * Float.BYTES;
        if (segment.byteSize() < expectedBytes) {
            throw new IllegalArgumentException(
                    "segment size %d < shape %s (expected %d bytes)"
                            .formatted(segment.byteSize(), Arrays.toString(shape), expectedBytes));
        }
    }

    public static Tensor of(final float[] data, final int... shape) {
        return new Tensor(MemorySegment.ofArray(data), shape);
    }

    /** Create tensor from float array, copying to native (off-heap) memory for zero-copy BLAS. */
    public static Tensor ofNative(final float[] data, final int... shape) {
        long bytes = (long) data.length * Float.BYTES;
        MemorySegment nat = allocSegment(bytes);
        MemorySegment.copy(data, 0, nat, FLOAT_LE, 0, data.length);
        return new Tensor(nat, shape);
    }

    public static Tensor ofSegment(final MemorySegment segment, final int... shape) {
        return new Tensor(segment, shape);
    }

    /** Allocate off-heap tensor, optionally zeroed. */
    public static Tensor zeros(final int... shape) {
        long bytes = (long) elementCount(shape) * Float.BYTES;
        MemorySegment seg = allocSegment(bytes);
        seg.fill((byte) 0);
        return new Tensor(seg, shape);
    }

    /** Allocate off-heap without zeroing (for scratch buffers we'll overwrite). */
    public static Tensor allocateNative(final int... shape) {
        long bytes = (long) elementCount(shape) * Float.BYTES;
        return new Tensor(allocSegment(bytes), shape);
    }

    /** Allocate off-heap bypassing scoped arena — for data that must outlive the current scope. */
    public static Tensor allocatePersistent(final int... shape) {
        long bytes = (long) elementCount(shape) * Float.BYTES;
        return new Tensor(Arena.ofAuto().allocate(bytes, Float.BYTES), shape);
    }

    /** Copy this tensor's data to an Arena.ofAuto() allocation so it survives scope close. */
    public Tensor copyToAutoArena() {
        long bytes = (long) this.size * Float.BYTES;
        MemorySegment dst = Arena.ofAuto().allocate(bytes, Float.BYTES);
        MemorySegment.copy(this.segment, 0, dst, 0, bytes);
        return new Tensor(dst, this.shape.clone());
    }

    /**
     * Set a scoped arena for the current thread. All allocateNative/ofNative/zeros calls
     * will use this arena instead of Arena.ofAuto(). Call {@link #clearScopedArena()} when done.
     * The arena is NOT closed by Tensor — caller owns the lifecycle.
     */
    public static void setScopedArena(final Arena arena) {
        SCOPED_ARENA.set(arena);
    }

    /** Clear the scoped arena for the current thread. */
    public static void clearScopedArena() {
        SCOPED_ARENA.remove();
    }

    private static MemorySegment allocSegment(final long bytes) {
        Arena arena = SCOPED_ARENA.get();
        if (arena != null) {
            return arena.allocate(bytes, Float.BYTES);
        }
        return Arena.ofAuto().allocate(bytes, Float.BYTES);
    }

    // ---- Accessors ----

    public MemorySegment segment() {
        return this.segment;
    }

    public int[] shape() {
        return this.shape;
    }

    public int rank() {
        return this.shape.length;
    }

    public int dim(final int i) {
        return this.shape[i < 0 ? this.shape.length + i : i];
    }

    public int size() {
        return this.size;
    }

    public boolean isNative() {
        return this.segment.isNative();
    }

    public float[] data() {
        float[] out = new float[this.size];
        MemorySegment.copy(this.segment, FLOAT_LE, 0, out, 0, this.size);
        return out;
    }

    /** Read a single row from the last dimension without copying the entire tensor. */
    public float[] getRow(final int lastDimIndex) {
        int lastDim = this.shape[this.shape.length - 1];
        float[] out = new float[lastDim];
        long byteOff = (long) lastDimIndex * lastDim * Float.BYTES;
        MemorySegment.copy(this.segment, FLOAT_LE, byteOff, out, 0, lastDim);
        return out;
    }

    public float get(final int... indices) {
        return this.segment.getAtIndex(FLOAT_LE, this.flatIndex(indices));
    }

    public void set(final float value, final int... indices) {
        this.segment.setAtIndex(FLOAT_LE, this.flatIndex(indices), value);
    }

    float getFlat(final int i) {
        return this.segment.getAtIndex(FLOAT_LE, i);
    }

    void setFlat(final int i, final float v) {
        this.segment.setAtIndex(FLOAT_LE, i, v);
    }

    // ---- Shape operations ----

    public Tensor reshape(final int... newShape) {
        int inferred = -1;
        int product = 1;
        int[] resolved = newShape;
        for (int i = 0; i < resolved.length; i++) {
            if (resolved[i] == -1) {
                if (inferred >= 0) { throw new IllegalArgumentException("only one -1 allowed"); }
                inferred = i;
            } else {
                product *= resolved[i];
            }
        }
        if (inferred >= 0) {
            resolved = resolved.clone();
            resolved[inferred] = this.size / product;
        }
        return new Tensor(this.segment, resolved);
    }

    public Tensor slice(final int from, final int to) {
        int rowSize = this.size / this.shape[0];
        long byteOff = (long) from * rowSize * Float.BYTES;
        long byteLen = (long) (to - from) * rowSize * Float.BYTES;
        return new Tensor(this.segment.asSlice(byteOff, byteLen), this.concatShape(to - from));
    }

    /** Transpose last two dimensions — direct segment-to-segment copy. */
    public Tensor transpose() {
        int rows = this.shape[this.shape.length - 2];
        int cols = this.shape[this.shape.length - 1];
        int batch = this.size / (rows * cols);
        Tensor out = Tensor.allocateNative(this.batchTransposedShape());
        for (int b = 0; b < batch; b++) {
            long srcBase = (long) b * rows * cols * Float.BYTES;
            long dstBase = (long) b * cols * rows * Float.BYTES;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    out.segment.set(FLOAT_LE, dstBase + ((long) c * rows + r) * Float.BYTES,
                            this.segment.get(FLOAT_LE, srcBase + ((long) r * cols + c) * Float.BYTES));
                }
            }
        }
        return out;
    }

    /** Transpose and ensure result is in native (off-heap) memory for zero-copy BLAS. */
    public Tensor transposeNative() {
        return this.transpose(); // transpose() already allocates native
    }

    /**
     * View into a pre-allocated (batchHeads, maxLen, headDim) buffer,
     * returning a tensor that acts as (batchHeads, viewLen, headDim).
     * The underlying memory is shared — no copy. Used for KV cache.
     */
    public Tensor viewRows(final int batchHeads, final int viewLen, final int headDim, final int maxLen) {
        if (viewLen == maxLen) {
            return this;
        }
        // Create a compact copy of just the active rows
        Tensor view = Tensor.allocateNative(batchHeads, viewLen, headDim);
        long rowBytes = (long) viewLen * headDim * Float.BYTES;
        long srcStride = (long) maxLen * headDim * Float.BYTES;
        for (int h = 0; h < batchHeads; h++) {
            MemorySegment.copy(this.segment, h * srcStride, view.segment, h * rowBytes, rowBytes);
        }
        return view;
    }

    // ---- SIMD-accelerated math ----

    public Tensor add(final Tensor other) {
        Tensor out = Tensor.allocateNative(this.shape.clone());
        SimdOps simd = SimdOps.get();
        if (other.size == this.size) {
            simd.add(this.segment, other.segment, out.segment, this.size);
        } else {
            int inner = other.size;
            for (int i = 0; i < this.size; i += inner) {
                long off = (long) i * Float.BYTES;
                simd.addOffset(this.segment, off, other.segment, 0, out.segment, off, inner);
            }
        }
        return out;
    }

    /** In-place add — avoids allocation when we own the tensor. */
    public Tensor addInPlace(final Tensor other) {
        SimdOps simd = SimdOps.get();
        if (other.size == this.size) {
            simd.add(this.segment, other.segment, this.segment, this.size);
        } else {
            int inner = other.size;
            for (int i = 0; i < this.size; i += inner) {
                long off = (long) i * Float.BYTES;
                simd.addOffset(this.segment, off, other.segment, 0, this.segment, off, inner);
            }
        }
        return this;
    }

    public Tensor scale(final float s) {
        Tensor out = Tensor.allocateNative(this.shape.clone());
        SimdOps.get().scale(this.segment, out.segment, this.size, s);
        return out;
    }

    /** Matrix multiply with fused scaling: alpha * (A @ B). Avoids separate scale allocation. */
    @SuppressWarnings({"checkstyle:LocalVariableName", "checkstyle:NestedForDepth"})
    public Tensor matmulScaled(final Tensor b, final float alpha) {
        int M = this.shape[this.shape.length - 2];
        int K = this.shape[this.shape.length - 1];
        int N = b.shape[b.shape.length - 1];
        if (b.shape[b.shape.length - 2] != K) {
            throw new IllegalArgumentException(
                    "matmul shape mismatch: K=%d vs %d".formatted(K, b.shape[b.shape.length - 2]));
        }
        int batch = this.size / (M * K);
        Tensor out = Tensor.allocateNative(this.matmulShape(M, N));

        if (M == 1 && !AccelerateBlas.isAvailable()) {
            boolean bBatched = b.size > N * K;
            for (int bi = 0; bi < batch; bi++) {
                long aOff = (long) bi * K * Float.BYTES;
                long bBase = bBatched ? (long) bi * N * K * Float.BYTES : 0;
                long cOff = (long) bi * N * Float.BYTES;
                for (int j = 0; j < N; j++) {
                    float dot = SimdOps.get().dot(this.segment, aOff,
                            b.segment, bBase + (long) j * K * Float.BYTES, K);
                    out.segment.set(FLOAT_LE, cOff + (long) j * Float.BYTES, dot * alpha);
                }
            }
            return out;
        }

        if (AccelerateBlas.isAvailable()) {
            this.matmulAccelerateScaled(b, out, batch, M, N, K, alpha);
        } else {
            this.matmulTiled(b, out, batch, M, N, K);
            if (alpha != 1.0f) {
                SimdOps.get().scaleInPlace(out.segment, out.size, alpha);
            }
        }
        return out;
    }

    /**
     * Matrix multiply with B transposed: A @ B^T, with optional scaling.
     * B shape: (..., N, K) — BLAS transposes internally, no allocation needed.
     * Result: (..., M, N).
     */
    @SuppressWarnings("checkstyle:LocalVariableName")
    public Tensor matmulTransB(final Tensor b, final float alpha) {
        int M = this.shape[this.shape.length - 2];
        int K = this.shape[this.shape.length - 1];
        int N = b.shape[b.shape.length - 2]; // B is (N, K), transposed to (K, N)
        if (b.shape[b.shape.length - 1] != K) {
            throw new IllegalArgumentException("matmulTransB shape mismatch");
        }
        int batch = this.size / (M * K);
        Tensor out = Tensor.allocateNative(this.matmulShape(M, N));

        // M=1 SIMD fallback when BLAS not available
        if (M == 1 && !AccelerateBlas.isAvailable()) {
            boolean bBatched = b.size > N * K;
            for (int bi = 0; bi < batch; bi++) {
                long aOff = (long) bi * K * Float.BYTES;
                long bBase = bBatched ? (long) bi * N * K * Float.BYTES : 0;
                long cOff = (long) bi * N * Float.BYTES;
                for (int j = 0; j < N; j++) {
                    float dot = SimdOps.get().dot(this.segment, aOff,
                            b.segment, bBase + (long) j * K * Float.BYTES, K);
                    out.segment.set(FLOAT_LE, cOff + (long) j * Float.BYTES, dot * alpha);
                }
            }
            return out;
        }

        if (AccelerateBlas.isAvailable()) {
            long aStride = (long) M * K * Float.BYTES;
            long bStride = (long) N * K * Float.BYTES;
            long cStride = (long) M * N * Float.BYTES;
            MemorySegment aSeg = ensureNative(this.segment, this.size);
            MemorySegment bSeg = ensureNative(b.segment, b.size);
            boolean bBatched = b.size > N * K;
            for (int bi = 0; bi < batch; bi++) {
                long bOff = bBatched ? bi * bStride : 0;
                AccelerateBlas.sgemmTransB(M, N, K,
                        alpha, aSeg.asSlice(bi * aStride, aStride), K,
                        bSeg.asSlice(bOff, bStride), K,
                        0.0f, out.segment.asSlice(bi * cStride, cStride), N);
            }
        } else {
            // Fallback: dot product of a rows with b rows (b is N×K, each row is a K-vector)
            int batchCount = this.size / (M * K);
            for (int bi = 0; bi < batchCount; bi++) {
                long aBase = (long) bi * M * K * Float.BYTES;
                long bBase = (long) bi * N * K * Float.BYTES;
                long cBase = (long) bi * M * N * Float.BYTES;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        float dot = SimdOps.get().dot(this.segment,
                                aBase + (long) i * K * Float.BYTES,
                                b.segment, bBase + (long) j * K * Float.BYTES, K);
                        out.segment.set(FLOAT_LE, cBase + ((long) i * N + j) * Float.BYTES, dot * alpha);
                    }
                }
            }
        }
        return out;
    }

    /**
     * Fast M=1 matrix-vector multiply using SIMD.
     * Computes out[j] = sum_k x[k] * B[k][j] by accumulating x[k] * B_row[k] into out.
     * B rows are contiguous in memory, enabling SIMD vectorization.
     * Avoids BLAS call overhead (~5μs/call × 66 calls/token = 330μs saved).
     */
    @SuppressWarnings("checkstyle:LocalVariableName")
    private void matmulVec(final Tensor b, final Tensor out,
                           final int batch, final int K, final int N) {
        boolean bBatched = b.size > K * N;
        SimdOps simd = SimdOps.get();
        for (int bi = 0; bi < batch; bi++) {
            long aOff = (long) bi * K * Float.BYTES;
            long bBase = bBatched ? (long) bi * K * N * Float.BYTES : 0;
            long cOff = (long) bi * N * Float.BYTES;
            for (int k = 0; k < K; k++) {
                float xk = this.segment.get(FLOAT_LE, aOff + (long) k * Float.BYTES);
                long bRowOff = bBase + (long) k * N * Float.BYTES;
                simd.matmulVecAccumulate(this.segment, aOff, xk, b.segment, bRowOff,
                        out.segment, cOff, N);
            }
        }
    }

    // ---- Matrix multiply ----

    @SuppressWarnings({"checkstyle:LocalVariableName", "checkstyle:NestedForDepth"})
    public Tensor matmul(final Tensor b) {
        int M = this.shape[this.shape.length - 2];
        int K = this.shape[this.shape.length - 1];
        int N = b.shape[b.shape.length - 1];
        if (b.shape[b.shape.length - 2] != K) {
            throw new IllegalArgumentException(
                    "matmul shape mismatch: K=%d vs %d".formatted(K, b.shape[b.shape.length - 2]));
        }
        int batch = this.size / (M * K);
        Tensor out = Tensor.allocateNative(this.matmulShape(M, N));

        if (M == 1 && !AccelerateBlas.isAvailable()) {
            this.matmulVec(b, out, batch, K, N);
        } else if (AccelerateBlas.isAvailable()) {
            this.matmulAccelerate(b, out, batch, M, N, K);
        } else {
            this.matmulTiled(b, out, batch, M, N, K);
        }
        return out;
    }

    @SuppressWarnings("checkstyle:LocalVariableName")
    private void matmulAccelerate(final Tensor b, final Tensor out,
                                  final int batch, final int M, final int N, final int K) {
        this.matmulAccelerateScaled(b, out, batch, M, N, K, 1.0f);
    }

    @SuppressWarnings("checkstyle:LocalVariableName")
    private void matmulAccelerateScaled(final Tensor b, final Tensor out,
                                        final int batch, final int M, final int N, final int K,
                                        final float alpha) {
        long aStride = (long) M * K * Float.BYTES;
        long bStride = (long) K * N * Float.BYTES;
        long cStride = (long) M * N * Float.BYTES;
        MemorySegment aSeg = ensureNative(this.segment, this.size);
        MemorySegment bSeg = ensureNative(b.segment, b.size);
        boolean bBatched = b.size > K * N; // B has batch dimension
        for (int bi = 0; bi < batch; bi++) {
            long bOff = bBatched ? bi * bStride : 0;
            AccelerateBlas.sgemm(M, N, K,
                    alpha, aSeg.asSlice(bi * aStride, aStride), K,
                    bSeg.asSlice(bOff, bStride), N,
                    0.0f, out.segment.asSlice(bi * cStride, cStride), N);
        }
    }

    static MemorySegment ensureNative(final MemorySegment seg, final int elements) {
        if (!seg.isNative()) {
            long bytes = (long) elements * Float.BYTES;
            MemorySegment nat = allocSegment(bytes);
            MemorySegment.copy(seg, 0, nat, 0, bytes);
            return nat;
        }
        return seg;
    }

    @SuppressWarnings({"checkstyle:LocalVariableName", "checkstyle:NestedForDepth"})
    private void matmulTiled(final Tensor b, final Tensor out,
                             final int batch, final int M, final int N, final int K) {
        final int tile = 64;
        boolean bBatched = b.size > K * N;
        for (int bi = 0; bi < batch; bi++) {
            long aBase = (long) bi * M * K * Float.BYTES;
            long bBase = bBatched ? (long) bi * K * N * Float.BYTES : 0;
            long cBase = (long) bi * M * N * Float.BYTES;
            for (int i0 = 0; i0 < M; i0 += tile) {
                int iEnd = Math.min(i0 + tile, M);
                for (int j0 = 0; j0 < N; j0 += tile) {
                    int jEnd = Math.min(j0 + tile, N);
                    for (int k0 = 0; k0 < K; k0 += tile) {
                        int kEnd = Math.min(k0 + tile, K);
                        for (int i = i0; i < iEnd; i++) {
                            for (int j = j0; j < jEnd; j++) {
                                float sum = k0 > 0
                                        ? out.segment.get(FLOAT_LE, cBase + ((long) i * N + j) * Float.BYTES) : 0;
                                for (int k = k0; k < kEnd; k++) {
                                    sum += this.segment.get(FLOAT_LE, aBase + ((long) i * K + k) * Float.BYTES)
                                            * b.segment.get(FLOAT_LE, bBase + ((long) k * N + j) * Float.BYTES);
                                }
                                out.segment.set(FLOAT_LE, cBase + ((long) i * N + j) * Float.BYTES, sum);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Activation functions ----

    private static float[] buildGeluTable() {
        float[] table = new float[65536];
        float s = (float) Math.sqrt(2.0 / Math.PI);
        for (int i = 0; i < 65536; i++) {
            float x = Float.float16ToFloat((short) i);
            table[i] = 0.5f * x * (1.0f + (float) Math.tanh(s * (x + 0.044715f * x * x * x)));
        }
        return table;
    }

    /** GELU activation using F16 lookup table (matching whisper.cpp ggml_vec_gelu_f32). */
    public Tensor gelu() {
        Tensor out = Tensor.allocateNative(this.shape.clone());
        int chunk = 4096;
        float[] buf = new float[chunk];
        float[] obuf = new float[chunk];
        for (int pos = 0; pos < this.size; pos += chunk) {
            int len = Math.min(chunk, this.size - pos);
            long byteOff = (long) pos * Float.BYTES;
            MemorySegment.copy(this.segment, FLOAT_LE, byteOff, buf, 0, len);
            for (int j = 0; j < len; j++) {
                // Convert to F16 index for table lookup
                short f16 = Float.floatToFloat16(buf[j]);
                obuf[j] = GELU_TABLE[Short.toUnsignedInt(f16)];
            }
            MemorySegment.copy(obuf, 0, out.segment, FLOAT_LE, byteOff, len);
        }
        return out;
    }

    /** In-place GELU — avoids allocation when we own the tensor. */
    public Tensor geluInPlace() {
        int chunk = 4096;
        float[] buf = new float[chunk];
        for (int pos = 0; pos < this.size; pos += chunk) {
            int len = Math.min(chunk, this.size - pos);
            long byteOff = (long) pos * Float.BYTES;
            MemorySegment.copy(this.segment, FLOAT_LE, byteOff, buf, 0, len);
            for (int j = 0; j < len; j++) {
                short f16 = Float.floatToFloat16(buf[j]);
                buf[j] = GELU_TABLE[Short.toUnsignedInt(f16)];
            }
            MemorySegment.copy(buf, 0, this.segment, FLOAT_LE, byteOff, len);
        }
        return this;
    }

    /** Softmax over the last dimension — in-place, uses vDSP on macOS. */
    public Tensor softmax() {
        int lastDim = this.shape[this.shape.length - 1];
        int outer = this.size / lastDim;

        if (AccelerateBlas.isAvailable() && this.segment.isNative()) {
            AccelerateBlas.softmaxRows(this.segment, outer, lastDim);
            return this;
        }

        // Fallback: bulk buffered with fastExp
        float[] row = new float[lastDim];
        for (int i = 0; i < outer; i++) {
            long byteOff = (long) i * lastDim * Float.BYTES;
            MemorySegment.copy(this.segment, FLOAT_LE, byteOff, row, 0, lastDim);
            float max = row[0];
            for (int j = 1; j < lastDim; j++) {
                if (row[j] > max) { max = row[j]; }
            }
            float sum = 0;
            for (int j = 0; j < lastDim; j++) {
                row[j] = (float) Math.exp(row[j] - max);
                sum += row[j];
            }
            float invSum = 1.0f / sum;
            for (int j = 0; j < lastDim; j++) { row[j] *= invSum; }
            MemorySegment.copy(row, 0, this.segment, FLOAT_LE, byteOff, lastDim);
        }
        return this;
    }

    /**
     * Fast exp approximation using Schraudolph's method with bias correction.
     * Accurate to ~0.1% relative error — sufficient for softmax normalization.
     */
    private static float fastExp(final float x) {
        if (x < -87f) { return 0f; }
        if (x > 88f) { return Float.MAX_VALUE; }
        // Schraudolph's method: reinterpret float bits
        // exp(x) ≈ 2^(x/ln2) via IEEE 754 bit manipulation
        final float a = 12102203.0f; // (1 << 23) / ln(2)
        final int b = 1065353216;     // 127 << 23 (IEEE bias)
        final int c = 60801;          // bias correction for better accuracy
        return Float.intBitsToFloat((int) (a * x) + b - c);
    }

    /** Layer normalization — bulk row processing. */
    public Tensor layerNorm(final Tensor gamma, final Tensor beta, final float eps) {
        return this.layerNorm(gamma, beta, eps, gamma.data(), beta.data());
    }

    /** Layer normalization with pre-cached gamma/beta arrays to avoid repeated off-heap copies. */
    public Tensor layerNorm(final Tensor gamma, final Tensor beta, final float eps,
                            final float[] gData, final float[] bData) {
        int lastDim = this.shape[this.shape.length - 1];
        int outer = this.size / lastDim;
        Tensor out = Tensor.allocateNative(this.shape.clone());
        float[] row = new float[lastDim];
        for (int i = 0; i < outer; i++) {
            long byteOff = (long) i * lastDim * Float.BYTES;
            MemorySegment.copy(this.segment, FLOAT_LE, byteOff, row, 0, lastDim);
            // mean
            float mean = 0;
            for (int j = 0; j < lastDim; j++) { mean += row[j]; }
            mean /= lastDim;
            // variance
            float variance = 0;
            for (int j = 0; j < lastDim; j++) {
                float d = row[j] - mean;
                variance += d * d;
            }
            float invStd = 1.0f / (float) Math.sqrt(variance / lastDim + eps);
            // normalize
            for (int j = 0; j < lastDim; j++) {
                row[j] = (row[j] - mean) * invStd * gData[j] + bData[j];
            }
            MemorySegment.copy(row, 0, out.segment, FLOAT_LE, byteOff, lastDim);
        }
        return out;
    }

    /** Add mask values in-place. mask is (maskRows, maskCols), applied to each batch slice. */
    public void addMaskInPlace(final Tensor mask, final int batchSlices, final int qLen, final int kvLen) {
        int maskCols = mask.dim(1);
        int maskRows = mask.dim(0);
        int applyQ = Math.min(qLen, maskRows);
        int applyK = Math.min(kvLen, maskCols);
        for (int h = 0; h < batchSlices; h++) {
            long hOff = (long) h * qLen * kvLen * Float.BYTES;
            for (int qi = 0; qi < applyQ; qi++) {
                long rowOff = hOff + (long) qi * kvLen * Float.BYTES;
                long maskRowOff = (long) qi * maskCols * Float.BYTES;
                for (int ki = 0; ki < applyK; ki++) {
                    long wIdx = rowOff + (long) ki * Float.BYTES;
                    long mIdx = maskRowOff + (long) ki * Float.BYTES;
                    this.segment.set(FLOAT_LE, wIdx,
                            this.segment.get(FLOAT_LE, wIdx) + mask.segment.get(FLOAT_LE, mIdx));
                }
            }
        }
    }

    // ---- Head rearrange (off-heap to off-heap) ----

    /** (batch, seq, nHead*headDim) → (batch*nHead, seq, headDim) — zero-copy segment ops. */
    public Tensor rearrangeToHeads(final int batch, final int seq, final int nHead, final int headDim) {
        Tensor dst = Tensor.allocateNative(batch * nHead, seq, headDim);
        long hdBytes = (long) headDim * Float.BYTES;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long srcRow = ((long) b * seq + s) * nHead * headDim * Float.BYTES;
                for (int h = 0; h < nHead; h++) {
                    long srcOff = srcRow + (long) h * headDim * Float.BYTES;
                    long dstOff = ((long) (b * nHead + h) * seq + s) * headDim * Float.BYTES;
                    MemorySegment.copy(this.segment, srcOff, dst.segment, dstOff, hdBytes);
                }
            }
        }
        return dst;
    }

    /** (batch*nHead, seq, headDim) → (batch, seq, nHead*headDim) — zero-copy segment ops. */
    public Tensor rearrangeFromHeads(final int batch, final int seq, final int nHead, final int headDim) {
        Tensor dst = Tensor.allocateNative(batch, seq, nHead * headDim);
        long hdBytes = (long) headDim * Float.BYTES;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long dstRow = ((long) b * seq + s) * nHead * headDim * Float.BYTES;
                for (int h = 0; h < nHead; h++) {
                    long srcOff = ((long) (b * nHead + h) * seq + s) * headDim * Float.BYTES;
                    long dstOff = dstRow + (long) h * headDim * Float.BYTES;
                    MemorySegment.copy(this.segment, srcOff, dst.segment, dstOff, hdBytes);
                }
            }
        }
        return dst;
    }

    // ---- SIMD primitives (delegated to SimdOps) ----

    static float simdDot(final MemorySegment a, final long aByteOff,
                         final MemorySegment b, final long bByteOff, final int len) {
        return SimdOps.get().dot(a, aByteOff, b, bByteOff, len);
    }

    /** Fused single-token attention per head — delegates to SimdOps. */
    @SuppressWarnings("checkstyle:ParameterNumber")
    public static void fusedAttentionHead(final MemorySegment qSeg, final long qOff,
                                          final MemorySegment kSeg, final long kvBase,
                                          final MemorySegment vSeg,
                                          final MemorySegment mSeg, final long outOff,
                                          final int kvLen, final int headDim, final float scale,
                                          final float[] scores) {
        SimdOps.get().fusedAttentionHead(qSeg, qOff, kSeg, kvBase, vSeg, mSeg, outOff,
                kvLen, headDim, scale, scores);
    }

    /** Returns true if the Vector API SIMD path is active. */
    public static boolean isVectorApiAvailable() {
        return SimdOps.isVectorApiAvailable();
    }


    // ---- Internal helpers ----

    private int flatIndex(final int[] indices) {
        int idx = 0;
        for (int i = 0; i < indices.length; i++) { idx += indices[i] * this.strides[i]; }
        return idx;
    }

    private int[] concatShape(final int first) {
        int[] ns = new int[this.shape.length];
        ns[0] = first;
        System.arraycopy(this.shape, 1, ns, 1, this.shape.length - 1);
        return ns;
    }

    private int[] batchTransposedShape() {
        int[] ns = this.shape.clone();
        ns[ns.length - 2] = this.shape[this.shape.length - 1];
        ns[ns.length - 1] = this.shape[this.shape.length - 2];
        return ns;
    }

    @SuppressWarnings("checkstyle:ParameterName")
    private int[] matmulShape(final int M, final int N) {
        int[] ns = this.shape.clone();
        ns[ns.length - 2] = M;
        ns[ns.length - 1] = N;
        return ns;
    }

    private static int[] computeStrides(final int[] shape) {
        int[] s = new int[shape.length];
        s[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) { s[i] = s[i + 1] * shape[i + 1]; }
        return s;
    }

    private static int elementCount(final int[] shape) {
        int n = 1;
        for (final int d : shape) { n *= d; }
        return n;
    }

    @Override
    public String toString() { return "Tensor" + Arrays.toString(this.shape); }
}

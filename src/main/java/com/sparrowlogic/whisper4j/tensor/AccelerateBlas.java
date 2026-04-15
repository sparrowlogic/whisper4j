package com.sparrowlogic.whisper4j.tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.logging.Logger;

/**
 * Apple Accelerate bindings via FFM: BLAS (cblas_sgemm) + vDSP + vForce (vvexpf).
 * On Apple Silicon, cblas_sgemm dispatches to the AMX coprocessor (~2400 GFLOPS).
 * vDSP/vForce provide SIMD-optimized element-wise math (exp, max, sum, etc).
 */
@SuppressWarnings({"checkstyle:ClassDataAbstractionCoupling", "checkstyle:ClassFanOutComplexity"})
final class AccelerateBlas {

    private static final Logger LOG = Logger.getLogger(AccelerateBlas.class.getName());
    private static final int ROW_MAJOR = 101;
    private static final int NO_TRANS = 111;
    private static final int TRANS = 112;

    private static final MethodHandle SGEMM;
    private static final MethodHandle VVEXPF;
    private static final MethodHandle MAXV;
    private static final MethodHandle VSADD;
    private static final MethodHandle SVE;
    private static final MethodHandle VSDIV;
    private static final MethodHandle SVDIV;
    private static final MethodHandle VSMUL;
    private static final MethodHandle VMUL;
    private static final boolean AVAILABLE;

    static {
        boolean avail = false;
        MethodHandle sgemm = null, vvexpf = null, maxv = null, vsadd = null;
        MethodHandle sve = null, vsdiv = null, svdiv = null, vsmul = null, vmul = null;
        try {
            var a = SymbolLookup.libraryLookup(
                    "/System/Library/Frameworks/Accelerate.framework/Accelerate", Arena.global());
            var l = Linker.nativeLinker();
            var A = ValueLayout.ADDRESS;
            var I = ValueLayout.JAVA_INT;
            var F = ValueLayout.JAVA_FLOAT;
            var L = ValueLayout.JAVA_LONG;

            sgemm = l.downcallHandle(a.find("cblas_sgemm").orElseThrow(),
                    FunctionDescriptor.ofVoid(I, I, I, I, I, I, F, A, I, A, I, F, A, I));
            vvexpf = l.downcallHandle(a.find("vvexpf").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, A, A));
            maxv = l.downcallHandle(a.find("vDSP_maxv").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, L, A, L));
            vsadd = l.downcallHandle(a.find("vDSP_vsadd").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, L, A, A, L, L));
            sve = l.downcallHandle(a.find("vDSP_sve").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, L, A, L));
            vsdiv = l.downcallHandle(a.find("vDSP_vsdiv").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, L, A, A, L, L));
            svdiv = l.downcallHandle(a.find("vDSP_svdiv").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, A, L, A, L, L));
            vsmul = l.downcallHandle(a.find("vDSP_vsmul").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, L, A, A, L, L));
            vmul = l.downcallHandle(a.find("vDSP_vmul").orElseThrow(),
                    FunctionDescriptor.ofVoid(A, L, A, L, A, L, L));
            avail = true;
            LOG.info("Apple Accelerate loaded (BLAS + vDSP + vForce)");
        } catch (final Throwable t) {
            LOG.fine("Accelerate not available: " + t.getMessage());
        }
        SGEMM = sgemm; VVEXPF = vvexpf; MAXV = maxv; VSADD = vsadd;
        SVE = sve; VSDIV = vsdiv; SVDIV = svdiv; VSMUL = vsmul; VMUL = vmul;
        AVAILABLE = avail;
    }

    private AccelerateBlas() { }

    static boolean isAvailable() { return AVAILABLE; }

    // ---- BLAS ----

    @SuppressWarnings("checkstyle:ParameterNumber")
    static void sgemm(int m, int n, int k, float alpha,
                      MemorySegment a, int lda, MemorySegment b, int ldb,
                      float beta, MemorySegment c, int ldc) {
        try {
            SGEMM.invokeExact(ROW_MAJOR, NO_TRANS, NO_TRANS, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } catch (final Throwable t) { throw new RuntimeException("sgemm failed", t); }
    }

    @SuppressWarnings("checkstyle:ParameterNumber")
    static void sgemmTransB(int m, int n, int k, float alpha,
                            MemorySegment a, int lda, MemorySegment b, int ldb,
                            float beta, MemorySegment c, int ldc) {
        try {
            SGEMM.invokeExact(ROW_MAJOR, NO_TRANS, TRANS, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } catch (final Throwable t) { throw new RuntimeException("sgemm transB failed", t); }
    }

    // ---- Parallel softmax using vDSP ----

    static void softmaxRows(MemorySegment seg, int rows, int cols) {
        int cores = Runtime.getRuntime().availableProcessors();
        int chunk = Math.max(1, rows / cores);
        Thread[] threads = new Thread[cores];
        for (int t = 0; t < cores; t++) {
            int start = t * chunk;
            int end = (t == cores - 1) ? rows : Math.min(start + chunk, rows);
            if (start >= rows) { break; }
            threads[t] = Thread.ofVirtual().start(() -> softmaxRowRange(seg, start, end, cols));
        }
        joinAll(threads);
    }

    private static void softmaxRowRange(MemorySegment seg, int startRow, int endRow, int cols) {
        long rowBytes = (long) cols * Float.BYTES;
        try (var arena = Arena.ofConfined()) {
            var maxB = arena.allocate(4L, 4);
            var negB = arena.allocate(4L, 4);
            var sumB = arena.allocate(4L, 4);
            var cntB = arena.allocate(4L, 4);
            cntB.set(ValueLayout.JAVA_INT, 0, cols);
            for (int r = startRow; r < endRow; r++) {
                var row = seg.asSlice((long) r * rowBytes, rowBytes);
                MAXV.invokeExact(row, 1L, maxB, (long) cols);
                negB.set(ValueLayout.JAVA_FLOAT, 0, -maxB.get(ValueLayout.JAVA_FLOAT, 0));
                VSADD.invokeExact(row, 1L, negB, row, 1L, (long) cols);
                VVEXPF.invokeExact(row, row, cntB);
                SVE.invokeExact(row, 1L, sumB, (long) cols);
                VSDIV.invokeExact(row, 1L, sumB, row, 1L, (long) cols);
            }
        } catch (final Throwable t) { throw new RuntimeException("vDSP softmax failed", t); }
    }

    // ---- Parallel GELU using vDSP: x * sigmoid(1.702x) ----

    static void geluInPlace(MemorySegment seg, int size) {
        int cores = Runtime.getRuntime().availableProcessors();
        int chunk = Math.max(1024, size / cores);
        int numChunks = (size + chunk - 1) / chunk;
        Thread[] threads = new Thread[numChunks];
        for (int t = 0; t < numChunks; t++) {
            int start = t * chunk;
            int len = Math.min(chunk, size - start);
            threads[t] = Thread.ofVirtual().start(() -> geluRange(seg, start, len));
        }
        joinAll(threads);
    }

    private static void geluRange(MemorySegment seg, int start, int len) {
        long off = (long) start * Float.BYTES;
        long bytes = (long) len * Float.BYTES;
        var slice = seg.asSlice(off, bytes);
        try (var arena = Arena.ofConfined()) {
            var tmp = arena.allocate(bytes, 4);
            var scalar = arena.allocate(4L, 4);
            var cnt = arena.allocate(4L, 4);
            cnt.set(ValueLayout.JAVA_INT, 0, len);
            // tmp = -1.702 * x
            scalar.set(ValueLayout.JAVA_FLOAT, 0, -1.702f);
            VSMUL.invokeExact(slice, 1L, scalar, tmp, 1L, (long) len);
            // tmp = exp(-1.702 * x)
            VVEXPF.invokeExact(tmp, tmp, cnt);
            // tmp = 1 + exp(-1.702 * x)
            scalar.set(ValueLayout.JAVA_FLOAT, 0, 1.0f);
            VSADD.invokeExact(tmp, 1L, scalar, tmp, 1L, (long) len);
            // tmp = 1 / (1 + exp(-1.702*x)) = sigmoid
            SVDIV.invokeExact(scalar, tmp, 1L, tmp, 1L, (long) len);
            // slice = x * sigmoid
            VMUL.invokeExact(slice, 1L, tmp, 1L, slice, 1L, (long) len);
        } catch (final Throwable t) { throw new RuntimeException("vDSP GELU failed", t); }
    }

    private static void joinAll(Thread[] threads) {
        for (var t : threads) {
            if (t != null) {
                try { t.join(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            }
        }
    }
}

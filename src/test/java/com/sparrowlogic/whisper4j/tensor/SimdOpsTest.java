package com.sparrowlogic.whisper4j.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Validates SimdOps detection and correctness on both scalar and SIMD paths.
 * Default surefire runs without --enable-preview (scalar path).
 * The -Pvector-api profile runs with --enable-preview (SIMD path).
 */
class SimdOpsTest {

    @Test
    void vectorApiDetectionMatchesRuntime() {
        boolean vectorModulePresent = isVectorModuleAvailable();
        assertEquals(vectorModulePresent, Tensor.isVectorApiAvailable(),
                "isVectorApiAvailable() should match whether jdk.incubator.vector is loadable");
    }

    @Test
    void dotProductScalarCorrectness() {
        float[] a = {1f, 2f, 3f, 4f};
        float[] b = {5f, 6f, 7f, 8f};
        var ta = Tensor.ofNative(a, 4);
        var tb = Tensor.ofNative(b, 4);
        float dot = Tensor.simdDot(ta.segment(), 0, tb.segment(), 0, 4);
        assertEquals(70f, dot, 1e-4f);
    }

    @Test
    void addDelegatesToCorrectPath() {
        var a = Tensor.ofNative(new float[]{1f, 2f, 3f}, 3);
        var b = Tensor.ofNative(new float[]{10f, 20f, 30f}, 3);
        var c = a.add(b);
        assertEquals(11f, c.data()[0], 1e-6f);
        assertEquals(22f, c.data()[1], 1e-6f);
        assertEquals(33f, c.data()[2], 1e-6f);
    }

    @Test
    void scaleDelegatesToCorrectPath() {
        var t = Tensor.ofNative(new float[]{2f, 4f, 6f}, 3);
        var s = t.scale(0.5f);
        assertEquals(1f, s.data()[0], 1e-6f);
        assertEquals(2f, s.data()[1], 1e-6f);
        assertEquals(3f, s.data()[2], 1e-6f);
    }

    @Test
    void largeDotProductMatchesBothPaths() {
        int n = 512;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.01f;
            b[i] = (n - i) * 0.01f;
        }
        var ta = Tensor.ofNative(a, n);
        var tb = Tensor.ofNative(b, n);
        float dot = Tensor.simdDot(ta.segment(), 0, tb.segment(), 0, n);
        // Expected: sum(i*(n-i)*0.0001) for i=0..511
        float expected = 0;
        for (int i = 0; i < n; i++) {
            expected += a[i] * b[i];
        }
        assertEquals(expected, dot, 0.1f);
    }

    @Test
    void vectorApiAvailableWithPreview() {
        if (isVectorModuleAvailable()) {
            assertTrue(Tensor.isVectorApiAvailable(),
                    "Vector API should be available when running with --enable-preview");
        }
    }

    @Test
    void vectorApiUnavailableWithoutPreview() {
        if (!isVectorModuleAvailable()) {
            assertFalse(Tensor.isVectorApiAvailable(),
                    "Vector API should not be available without --enable-preview");
        }
    }

    private static boolean isVectorModuleAvailable() {
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            return true;
        } catch (final Throwable t) {
            return false;
        }
    }
}

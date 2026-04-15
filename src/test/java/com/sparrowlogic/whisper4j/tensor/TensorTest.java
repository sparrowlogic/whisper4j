package com.sparrowlogic.whisper4j.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorTest {

    @Test
    void matmul2x3Times3x2() {
        // numpy: [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        var a = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        var b = Tensor.of(new float[]{7, 8, 9, 10, 11, 12}, 3, 2);
        var c = a.matmul(b);
        assertArrayEquals(new int[]{2, 2}, c.shape());
        assertEquals(58, c.get(0, 0), 1e-4);
        assertEquals(64, c.get(0, 1), 1e-4);
        assertEquals(139, c.get(1, 0), 1e-4);
        assertEquals(154, c.get(1, 1), 1e-4);
    }

    @Test
    void matmulIdentity() {
        var a = Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        var eye = Tensor.of(new float[]{1, 0, 0, 1}, 2, 2);
        var c = a.matmul(eye);
        assertArrayEquals(a.data(), c.data(), 1e-5f);
    }

    @Test
    void matmulLargeVectorized() {
        int n = 128;
        float[] ad = new float[n * n];
        float[] bd = new float[n * n];
        for (int i = 0; i < n * n; i++) { ad[i] = i * 0.001f; bd[i] = (n * n - i) * 0.001f; }
        var a = Tensor.of(ad, n, n);
        var b = Tensor.of(bd, n, n);
        var c = a.matmul(b);
        assertTrue(c.get(0, 0) > 0);
    }

    @Test
    void geluZeroIsZero() {
        var t = Tensor.of(new float[]{0}, 1);
        assertEquals(0, t.gelu().data()[0], 1e-5);
    }

    @Test
    void geluPositiveIsPositive() {
        var t = Tensor.of(new float[]{1.0f}, 1);
        assertTrue(t.gelu().data()[0] > 0);
    }

    @Test
    void geluNegativeNearZero() {
        var t = Tensor.of(new float[]{-3.0f}, 1);
        assertTrue(Math.abs(t.gelu().data()[0]) < 0.01);
    }

    @Test
    void softmaxSumsToOne() {
        var t = Tensor.of(new float[]{1, 2, 3, 4}, 1, 4);
        var s = t.softmax();
        float sum = 0;
        for (float v : s.data()) { sum += v; }
        assertEquals(1.0f, sum, 1e-5f);
    }

    @Test
    void softmaxMaxElementHasHighestProb() {
        var t = Tensor.of(new float[]{1, 5, 2}, 1, 3);
        var s = t.softmax();
        assertTrue(s.data()[1] > s.data()[0]);
        assertTrue(s.data()[1] > s.data()[2]);
    }

    @Test
    void softmaxMultipleRows() {
        var t = Tensor.of(new float[]{1, 2, 3, 10, 20, 30}, 2, 3);
        var s = t.softmax();
        float sum0 = s.data()[0] + s.data()[1] + s.data()[2];
        float sum1 = s.data()[3] + s.data()[4] + s.data()[5];
        assertEquals(1.0f, sum0, 1e-4f);
        assertEquals(1.0f, sum1, 1e-4f);
    }

    @Test
    void layerNormNormalizesOutput() {
        var x = Tensor.of(new float[]{1, 2, 3, 4}, 1, 4);
        var g = Tensor.of(new float[]{1, 1, 1, 1}, 4);
        var b = Tensor.of(new float[]{0, 0, 0, 0}, 4);
        var out = x.layerNorm(g, b, 1e-5f);
        float mean = 0;
        for (float v : out.data()) { mean += v; }
        mean /= 4;
        assertEquals(0, mean, 1e-4);
    }

    @Test
    void layerNormWithScaleAndShift() {
        var x = Tensor.of(new float[]{0, 0, 0, 0}, 1, 4);
        var g = Tensor.of(new float[]{2, 2, 2, 2}, 4);
        var b = Tensor.of(new float[]{1, 1, 1, 1}, 4);
        var out = x.layerNorm(g, b, 1e-5f);
        for (float v : out.data()) { assertEquals(1.0f, v, 1e-4f); }
    }

    @Test
    void addElementwise() {
        var a = Tensor.of(new float[]{1, 2, 3}, 3);
        var b = Tensor.of(new float[]{10, 20, 30}, 3);
        var c = a.add(b);
        assertArrayEquals(new float[]{11, 22, 33}, c.data(), 1e-6f);
    }

    @Test
    void addBroadcastBias() {
        var a = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        var b = Tensor.of(new float[]{10, 20, 30}, 3);
        var c = a.add(b);
        assertArrayEquals(new float[]{11, 22, 33, 14, 25, 36}, c.data(), 1e-6f);
    }

    @Test
    void scaleMultiplies() {
        var t = Tensor.of(new float[]{1, 2, 3}, 3);
        var s = t.scale(2.0f);
        assertArrayEquals(new float[]{2, 4, 6}, s.data(), 1e-6f);
    }

    @Test
    void reshapePreservesData() {
        var t = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        var r = t.reshape(3, 2);
        assertArrayEquals(new int[]{3, 2}, r.shape());
        assertEquals(1f, r.get(0, 0), 1e-6f);
        assertEquals(6f, r.get(2, 1), 1e-6f);
    }

    @Test
    void reshapeInferredDimension() {
        var t = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 6);
        var r = t.reshape(2, 3);
        assertArrayEquals(new int[]{2, 3}, r.shape());
    }

    @Test
    void sliceRows() {
        var t = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 3, 2);
        var s = t.slice(1, 2);
        assertArrayEquals(new int[]{1, 2}, s.shape());
        assertArrayEquals(new float[]{3, 4}, s.data(), 1e-6f);
    }

    @Test
    void transpose2d() {
        var t = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        var tr = t.transpose();
        assertArrayEquals(new int[]{3, 2}, tr.shape());
        assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, tr.data(), 1e-6f);
    }

    @Test
    void getAndSet() {
        var t = Tensor.zeros(2, 3);
        t.set(42f, 1, 2);
        assertEquals(42f, t.get(1, 2), 1e-6f);
        assertEquals(0f, t.get(0, 0), 1e-6f);
    }

    @Test
    void shapeMismatchThrows() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.of(new float[5], 2, 3));
    }

    @Test
    void geluInPlaceMatchesGelu() {
        float[] data = {-2f, -1f, 0f, 1f, 2f};
        var g1 = Tensor.of(data.clone(), 1, 5).gelu();
        var g2 = Tensor.of(data.clone(), 1, 5).geluInPlace();
        assertArrayEquals(g1.data(), g2.data(), 1e-5f);
    }

    @Test
    void sliceRange() {
        var t = Tensor.of(new float[]{10, 20, 30, 40, 50}, 5, 1);
        var s = t.slice(1, 3);
        assertArrayEquals(new int[]{2, 1}, s.shape());
        assertArrayEquals(new float[]{20, 30}, s.data(), 1e-6f);
    }

    @Test
    void getRowReturnsLastDim() {
        var t = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertArrayEquals(new float[]{4, 5, 6}, t.getRow(1), 1e-6f);
    }

    @Test
    void zerosAllZero() {
        var t = Tensor.zeros(2, 3);
        for (float v : t.data()) { assertEquals(0f, v); }
    }

    @Test
    void rankAndDim() {
        var t = Tensor.of(new float[6], 2, 3);
        assertEquals(2, t.rank());
        assertEquals(2, t.dim(0));
        assertEquals(3, t.dim(1));
        assertEquals(6, t.size());
    }

    @Test
    void addInPlaceSameSize() {
        var a = Tensor.of(new float[]{1, 2, 3}, 3);
        a.addInPlace(Tensor.of(new float[]{10, 20, 30}, 3));
        assertArrayEquals(new float[]{11, 22, 33}, a.data(), 1e-6f);
    }

    // ---- Python numpy-verified tests ----

    @Test
    void matmulTransBEquivalent() {
        // numpy: [[1,2],[3,4]] @ [[5,6],[7,8]]^T = [[17,23],[39,53]]
        var a = Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        var b = Tensor.of(new float[]{5, 6, 7, 8}, 2, 2);
        var c = a.matmulTransB(b, 1.0f);
        assertEquals(17f, c.get(0, 0), 1e-4f);
        assertEquals(23f, c.get(0, 1), 1e-4f);
        assertEquals(39f, c.get(1, 0), 1e-4f);
        assertEquals(53f, c.get(1, 1), 1e-4f);
    }

    @Test
    void matmulTransBNonSquare() {
        // numpy: (2,3) @ (2,3)^T = [[14,32],[32,77]]
        var a = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        var b = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        var c = a.matmulTransB(b, 1.0f);
        assertEquals(14f, c.get(0, 0), 1e-4f);
        assertEquals(32f, c.get(0, 1), 1e-4f);
        assertEquals(32f, c.get(1, 0), 1e-4f);
        assertEquals(77f, c.get(1, 1), 1e-4f);
    }

    @Test
    void matmulTransBM1() {
        // numpy: (1,4) @ (3,4)^T = [[1,2,3]]
        var a = Tensor.of(new float[]{1, 2, 3, 4}, 1, 4);
        var b = Tensor.of(new float[]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 3, 4);
        var c = a.matmulTransB(b, 1.0f);
        assertArrayEquals(new int[]{1, 3}, c.shape());
        assertEquals(1f, c.get(0, 0), 1e-4f);
        assertEquals(2f, c.get(0, 1), 1e-4f);
        assertEquals(3f, c.get(0, 2), 1e-4f);
    }

    @Test
    void matmulScaledVerified() {
        // numpy: 0.5 * [[1,2],[3,4]] @ [[5,6],[7,8]] = [[9.5,11],[21.5,25]]
        var a = Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        var b = Tensor.of(new float[]{5, 6, 7, 8}, 2, 2);
        var c = a.matmulScaled(b, 0.5f);
        assertEquals(9.5f, c.get(0, 0), 1e-4f);
        assertEquals(11f, c.get(0, 1), 1e-4f);
        assertEquals(21.5f, c.get(1, 0), 1e-4f);
        assertEquals(25f, c.get(1, 1), 1e-4f);
    }

    @Test
    void softmaxVerified() {
        // numpy: softmax([1,2,3,4]) = [0.0321, 0.0871, 0.2369, 0.6439]
        var s = Tensor.of(new float[]{1, 2, 3, 4}, 1, 4).softmax();
        assertEquals(0.0321f, s.get(0, 0), 1e-3f);
        assertEquals(0.0871f, s.get(0, 1), 1e-3f);
        assertEquals(0.2369f, s.get(0, 2), 1e-3f);
        assertEquals(0.6439f, s.get(0, 3), 1e-3f);
    }

    @Test
    void layerNormVerified() {
        // numpy: layerNorm([1,2,3,4,5]) = [-1.4142, -0.7071, 0, 0.7071, 1.4142]
        var x = Tensor.of(new float[]{1, 2, 3, 4, 5}, 1, 5);
        var g = Tensor.of(new float[]{1, 1, 1, 1, 1}, 5);
        var b = Tensor.of(new float[]{0, 0, 0, 0, 0}, 5);
        var out = x.layerNorm(g, b, 1e-5f);
        assertEquals(-1.4142f, out.get(0, 0), 1e-3f);
        assertEquals(0f, out.get(0, 2), 1e-3f);
        assertEquals(1.4142f, out.get(0, 4), 1e-3f);
    }

    @Test
    void geluVerified() {
        // numpy: gelu([-2,-1,0,1,2]) = [-0.0454, -0.1588, 0, 0.8412, 1.9546]
        var g = Tensor.of(new float[]{-2, -1, 0, 1, 2}, 1, 5).gelu();
        assertEquals(-0.0454f, g.get(0, 0), 0.01f);
        assertEquals(-0.1588f, g.get(0, 1), 0.01f);
        assertEquals(0f, g.get(0, 2), 0.01f);
        assertEquals(0.8412f, g.get(0, 3), 0.01f);
        assertEquals(1.9546f, g.get(0, 4), 0.01f);
    }

    @Test
    void batchedMatmulVerified() {
        // numpy: (2,2,3) @ (2,3,2) → batch0=[[4,2],[10,5]], batch1=[[8,16],[11,22]]
        var a = Tensor.of(new float[]{1,2,3, 4,5,6, 7,8,9, 10,11,12}, 2, 2, 3);
        var b = Tensor.of(new float[]{1,0, 0,1, 1,0, 0,1, 1,0, 0,1}, 2, 3, 2);
        var c = a.matmul(b);
        assertArrayEquals(new int[]{2, 2, 2}, c.shape());
        assertEquals(4f, c.get(0, 0, 0), 1e-4f);
        assertEquals(2f, c.get(0, 0, 1), 1e-4f);
        assertEquals(10f, c.get(0, 1, 0), 1e-4f);
        assertEquals(5f, c.get(0, 1, 1), 1e-4f);
        assertEquals(8f, c.get(1, 0, 0), 1e-4f);
        assertEquals(16f, c.get(1, 0, 1), 1e-4f);
        assertEquals(11f, c.get(1, 1, 0), 1e-4f);
        assertEquals(22f, c.get(1, 1, 1), 1e-4f);
    }
}

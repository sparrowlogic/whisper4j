package com.sparrowlogic.whisper4j;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class AlignmentTest {

    @Test
    void emptyTokensReturnsEmpty() {
        var result = Alignment.findAlignment(
                List.of(Tensor.zeros(1, 5, 10)),
                new int[0], null, 20, 3, new float[0]);
        assertTrue(result.isEmpty());
    }

    @Test
    void emptyWeightsReturnsEmpty() {
        var result = Alignment.findAlignment(
                List.of(), new int[]{100, 200}, null, 20, 3, new float[]{-0.5f, -0.3f});
        assertTrue(result.isEmpty());
    }

    @Test
    void zeroTextLenReturnsEmpty() {
        // sotSeqLen >= seqLen means textLen <= 0
        var result = Alignment.findAlignment(
                List.of(Tensor.zeros(1, 3, 10)),
                new int[]{100}, null, 20, 5, new float[]{-0.5f});
        assertTrue(result.isEmpty());
    }

    @Test
    void wordTimingRecord() {
        var wt = new Alignment.WordTiming("hello", new int[]{100}, 0.5f, 1.0f, 0.9f);
        assertEquals("hello", wt.word());
        assertArrayEquals(new int[]{100}, wt.tokens());
        assertEquals(0.5f, wt.start());
        assertEquals(1.0f, wt.end());
        assertEquals(0.9f, wt.probability());
    }

    @Test
    void dtwIdentityPath() {
        // 3x3 identity-like cost: diagonal should be cheapest
        float[] cost = {
            0, 1, 1,
            1, 0, 1,
            1, 1, 0
        };
        int[][] path = Alignment.dtw(cost, 3, 3);
        int[] textIdx = path[0];
        int[] timeIdx = path[1];
        // Path should follow diagonal
        assertTrue(textIdx.length >= 3);
        assertEquals(0, textIdx[0]);
        assertEquals(2, textIdx[textIdx.length - 1]);
    }

    @Test
    void dtwSingleElement() {
        float[] cost = {5.0f};
        int[][] path = Alignment.dtw(cost, 1, 1);
        assertEquals(1, path[0].length);
        assertEquals(0, path[0][0]);
        assertEquals(0, path[1][0]);
    }

    @Test
    void dtwRectangular() {
        // 2 tokens, 4 time steps
        float[] cost = {
            0, 1, 1, 1,
            1, 1, 1, 0
        };
        int[][] path = Alignment.dtw(cost, 2, 4);
        assertTrue(path[0].length >= 2);
        // First token should align to early frames, second to late
        assertEquals(0, path[0][0]);
        assertEquals(1, path[0][path[0].length - 1]);
    }

    @Test
    void findAlignmentZeroNumFrames() {
        // numFrames=0 means kvLen=0, should return empty
        var result = Alignment.findAlignment(
                List.of(Tensor.zeros(1, 5, 10)),
                new int[]{100}, null, 0, 3, new float[]{-0.5f});
        assertTrue(result.isEmpty());
    }
}

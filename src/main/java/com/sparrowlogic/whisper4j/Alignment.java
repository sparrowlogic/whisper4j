package com.sparrowlogic.whisper4j;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * DTW alignment and word timestamp extraction.
 * Ported from openai/whisper (cba3768) whisper/timing.py.
 */
public final class Alignment {

    private static final float TOKENS_PER_SECOND = 50.0f;

    private Alignment() { }

    /**
     * Find word-level alignment using cross-attention weights and DTW.
     *
     * @param crossAttnWeights list of attention weight tensors per alignment head
     *                         each (nHead, tokenLen, audioFrames)
     * @param textTokens       the generated text tokens (no special tokens)
     * @param tokenizer        for splitting tokens to words
     * @param numFrames        number of audio frames in this segment
     * @return list of word timings
     */
    @SuppressWarnings({"checkstyle:ExecutableStatementCount", "checkstyle:CyclomaticComplexity",
        "checkstyle:NPathComplexity", "checkstyle:ReturnCount"})
    public static List<WordTiming> findAlignment(
            final List<Tensor> crossAttnWeights,
            final int[] textTokens,
            final com.sparrowlogic.whisper4j.tokenizer.WhisperTokenizer tokenizer,
            final int numFrames,
            final int sotSeqLen,
            final float[] tokenProbs) {

        if (textTokens.length == 0 || crossAttnWeights.isEmpty()) {
            return List.of();
        }

        // Average attention weights across alignment heads
        // Each weight tensor is (batchHeads, seqLen, kvLen)
        // We want the portion corresponding to text tokens (after SOT sequence)
        Tensor first = crossAttnWeights.getFirst();
        int seqLen = first.dim(1);
        int kvLen = Math.min(first.dim(2), numFrames / 2);

        int textStart = sotSeqLen;
        int textEnd = Math.min(textStart + textTokens.length, seqLen);
        int textLen = textEnd - textStart;
        if (textLen <= 0 || kvLen <= 0) {
            return List.of();
        }

        // Average across all heads and layers, extract text token rows
        float[] matrix = new float[textLen * kvLen];
        int headCount = 0;
        for (Tensor w : crossAttnWeights) {
            float[] wd = w.data();
            int wSeq = w.dim(1);
            int wKv = w.dim(2);
            int nHeads = w.dim(0);
            for (int h = 0; h < nHeads; h++) {
                for (int t = 0; t < textLen; t++) {
                    int srcRow = textStart + t;
                    if (srcRow >= wSeq) {
                        continue;
                    }
                    for (int f = 0; f < kvLen; f++) {
                        if (f < wKv) {
                            matrix[t * kvLen + f] += wd[h * wSeq * wKv + srcRow * wKv + f];
                        }
                    }
                }
                headCount++;
            }
        }
        // Normalize
        if (headCount > 0) {
            float inv = 1.0f / headCount;
            for (int i = 0; i < matrix.length; i++) {
                matrix[i] *= inv;
            }
        }

        // Apply median filter (width=7) along time axis per token
        medianFilterRows(matrix, textLen, kvLen, 7);

        // DTW on negated matrix
        float[] negMatrix = new float[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            negMatrix[i] = -matrix[i];
        }
        int[][] path = dtw(negMatrix, textLen, kvLen);
        int[] textIndices = path[0];
        int[] timeIndices = path[1];

        // Find jump points (where text index changes)
        float[] jumpTimes = extractJumpTimes(textIndices, timeIndices);

        // Split tokens into words and compute boundaries
        return buildWordTimings(textTokens, tokenizer, jumpTimes, tokenProbs);
    }

    /** DTW on cost matrix (textLen x kvLen). Returns [textIndices, timeIndices]. */
    static int[][] dtw(final float[] x, final int n, final int m) {
        float[] cost = new float[(n + 1) * (m + 1)];
        int[] trace = new int[(n + 1) * (m + 1)];
        Arrays.fill(cost, Float.MAX_VALUE);
        Arrays.fill(trace, -1);
        cost[0] = 0;

        for (int j = 1; j <= m; j++) {
            for (int i = 1; i <= n; i++) {
                float c0 = cost[(i - 1) * (m + 1) + (j - 1)]; // diagonal
                float c1 = cost[(i - 1) * (m + 1) + j];        // up
                float c2 = cost[i * (m + 1) + (j - 1)];        // left
                float c;
                int t;
                if (c0 <= c1 && c0 <= c2) {
                    c = c0; t = 0;
                } else if (c1 <= c2) {
                    c = c1; t = 1;
                } else {
                    c = c2; t = 2;
                }
                cost[i * (m + 1) + j] = x[(i - 1) * m + (j - 1)] + c;
                trace[i * (m + 1) + j] = t;
            }
        }

        return backtrace(trace, n, m);
    }

    private static int[][] backtrace(final int[] trace, final int n, final int m) {
        int stride = m + 1;
        // Set boundary conditions
        for (int j = 0; j <= m; j++) {
            trace[j] = 2; // first row: left
        }
        for (int i = 0; i <= n; i++) {
            trace[i * stride] = 1; // first col: up
        }

        List<int[]> result = new ArrayList<>();
        int i = n;
        int j = m;
        while (i > 0 || j > 0) {
            result.add(new int[]{i - 1, j - 1});
            int t = trace[i * stride + j];
            if (t == 0) {
                i--; j--;
            } else if (t == 1) {
                i--;
            } else {
                j--;
            }
        }

        int[] textIdx = new int[result.size()];
        int[] timeIdx = new int[result.size()];
        for (int k = 0; k < result.size(); k++) {
            int[] pair = result.get(result.size() - 1 - k);
            textIdx[k] = pair[0];
            timeIdx[k] = pair[1];
        }
        return new int[][]{textIdx, timeIdx};
    }

    private static float[] extractJumpTimes(final int[] textIndices, final int[] timeIndices) {
        List<Float> jumps = new ArrayList<>();
        int prev = -1;
        for (int k = 0; k < textIndices.length; k++) {
            if (textIndices[k] != prev) {
                jumps.add(timeIndices[k] / TOKENS_PER_SECOND);
                prev = textIndices[k];
            }
        }
        float[] result = new float[jumps.size()];
        for (int i = 0; i < jumps.size(); i++) {
            result[i] = jumps.get(i);
        }
        return result;
    }

    @SuppressWarnings("checkstyle:ExecutableStatementCount")
    private static List<WordTiming> buildWordTimings(
            final int[] textTokens,
            final com.sparrowlogic.whisper4j.tokenizer.WhisperTokenizer tokenizer,
            final float[] jumpTimes,
            final float[] tokenProbs) {

        // Split tokens into words using the tokenizer's decode
        List<WordTiming> words = new ArrayList<>();
        StringBuilder currentWord = new StringBuilder();
        List<Integer> currentTokens = new ArrayList<>();
        int tokenStart = 0;
        float probSum = 0;

        for (int i = 0; i < textTokens.length; i++) {
            String decoded = tokenizer.decode(new int[]{textTokens[i]});
            boolean startsNewWord = decoded.startsWith(" ") && !currentWord.isEmpty();

            if (startsNewWord) {
                float start = tokenStart < jumpTimes.length ? jumpTimes[tokenStart] : 0;
                float end = i < jumpTimes.length ? jumpTimes[i] : start;
                float prob = currentTokens.isEmpty() ? 0 : probSum / currentTokens.size();
                words.add(new WordTiming(currentWord.toString(),
                        currentTokens.stream().mapToInt(Integer::intValue).toArray(),
                        start, end, prob));
                currentWord = new StringBuilder();
                currentTokens = new ArrayList<>();
                tokenStart = i;
                probSum = 0;
            }

            currentWord.append(decoded);
            currentTokens.add(textTokens[i]);
            if (tokenProbs != null && i < tokenProbs.length) {
                probSum += tokenProbs[i];
            }
        }

        if (!currentWord.isEmpty()) {
            float start = tokenStart < jumpTimes.length ? jumpTimes[tokenStart] : 0;
            float end = jumpTimes.length > 0 ? jumpTimes[jumpTimes.length - 1] : start;
            float prob = currentTokens.isEmpty() ? 0 : probSum / currentTokens.size();
            words.add(new WordTiming(currentWord.toString(),
                    currentTokens.stream().mapToInt(Integer::intValue).toArray(),
                    start, end, prob));
        }

        return words;
    }

    /** In-place median filter along columns (time axis) for each row (token). */
    private static void medianFilterRows(final float[] matrix, final int rows,
                                         final int cols, final int width) {
        int pad = width / 2;
        float[] buf = new float[width];
        float[] row = new float[cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(matrix, r * cols, row, 0, cols);
            for (int c = 0; c < cols; c++) {
                int count = 0;
                for (int k = -pad; k <= pad; k++) {
                    int idx = c + k;
                    if (idx < 0) {
                        idx = -idx;
                    }
                    if (idx >= cols) {
                        idx = 2 * cols - idx - 2;
                    }
                    if (idx >= 0 && idx < cols) {
                        buf[count++] = row[idx];
                    }
                }
                Arrays.sort(buf, 0, count);
                matrix[r * cols + c] = buf[count / 2];
            }
        }
    }

    /** Word timing result. */
    public record WordTiming(String word, int[] tokens, float start, float end, float probability) { }
}

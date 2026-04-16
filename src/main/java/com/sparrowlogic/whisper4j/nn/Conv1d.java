package com.sparrowlogic.whisper4j.nn;

import com.sparrowlogic.whisper4j.annotation.Nullable;
import com.sparrowlogic.whisper4j.tensor.Tensor;

/**
 * 1D convolution via im2col + BLAS matmul.
 * weight: (outCh, inCh, kSize), input: (batch, inCh, length), output: (batch, outCh, outLen)
 */
public final class Conv1d {

    private final int outCh;
    private final int inCh;
    private final int kSize;
    private final int stride;
    private final int padding;
    private final Tensor weightMatrix;
    private final @Nullable Tensor bias;

    public Conv1d(final Tensor weight, final @Nullable Tensor bias,
                  final int stride, final int padding) {
        this.stride = stride;
        this.padding = padding;
        this.outCh = weight.dim(0);
        this.inCh = weight.dim(1);
        this.kSize = weight.dim(2);
        this.weightMatrix = weight.reshape(this.outCh, this.inCh * this.kSize);
        this.bias = (bias != null && bias.rank() > 1) ? bias.reshape(bias.size()) : bias;
    }

    @SuppressWarnings("checkstyle:NestedForDepth")
    /**
     * Apply 1D convolution via im2col + BLAS matmul.
     *
     * @param input tensor of shape (batch, inChannels, length)
     * @return tensor of shape (batch, outChannels, outLength)
     */
    public Tensor forward(final Tensor input) {
        int inLen = input.dim(2);
        int outLen = (inLen + 2 * this.padding - this.kSize) / this.stride + 1;
        int colRows = this.inCh * this.kSize;

        Tensor col = this.buildIm2Col(input.data(), inLen, outLen, colRows);
        Tensor result = this.weightMatrix.matmul(col);
        result = this.addBias(result, outLen);
        return result.reshape(1, this.outCh, outLen);
    }

    @SuppressWarnings("checkstyle:NestedForDepth")
    private Tensor buildIm2Col(final float[] inData, final int inLen, final int outLen, final int colRows) {
        float[] colData = new float[colRows * outLen];
        for (int t = 0; t < outLen; t++) {
            int inStart = t * this.stride - this.padding;
            for (int ic = 0; ic < this.inCh; ic++) {
                int inBase = ic * inLen;
                for (int k = 0; k < this.kSize; k++) {
                    int inIdx = inStart + k;
                    colData[(ic * this.kSize + k) * outLen + t] =
                            (inIdx >= 0 && inIdx < inLen) ? inData[inBase + inIdx] : 0;
                }
            }
        }
        return Tensor.ofNative(colData, colRows, outLen);
    }

    private Tensor addBias(final Tensor result, final int outLen) {
        if (this.bias == null) {
            return result;
        }
        float[] rd = result.data();
        float[] bd = this.bias.data();
        for (int oc = 0; oc < this.outCh; oc++) {
            float bv = bd[oc];
            int rowOff = oc * outLen;
            for (int t = 0; t < outLen; t++) {
                rd[rowOff + t] += bv;
            }
        }
        return Tensor.ofNative(rd, this.outCh, outLen);
    }
}

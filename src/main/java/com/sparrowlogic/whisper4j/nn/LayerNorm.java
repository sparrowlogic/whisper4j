package com.sparrowlogic.whisper4j.nn;

import com.sparrowlogic.whisper4j.tensor.Tensor;

/** Layer normalization over the last dimension. */
public final class LayerNorm {

    private final Tensor weight;
    private final Tensor bias;
    private final float eps;
    /** Pre-cached gamma/beta arrays to avoid off-heap copy on every forward call. */
    private final float[] gammaData;
    private final float[] betaData;

    public LayerNorm(final Tensor weight, final Tensor bias) {
        this(weight, bias, 1e-5f);
    }

    public LayerNorm(final Tensor weight, final Tensor bias, final float eps) {
        this.weight = weight;
        this.bias = bias;
        this.eps = eps;
        this.gammaData = weight.data();
        this.betaData = bias.data();
    }

    public Tensor forward(final Tensor x) {
        return x.layerNorm(this.weight, this.bias, this.eps, this.gammaData, this.betaData);
    }
}

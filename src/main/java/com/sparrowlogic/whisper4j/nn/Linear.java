package com.sparrowlogic.whisper4j.nn;

import com.sparrowlogic.whisper4j.tensor.Tensor;

/**
 * y = x @ W^T + b.
 * Weight is pre-transposed at construction time for optimal BLAS performance.
 * This avoids the CblasTrans flag overhead on every forward call.
 */
public final class Linear {
    /** Pre-transposed weight: (inFeatures, outFeatures) for zero-copy BLAS matmul. */
    private final Tensor weightT;
    private final Tensor bias;

    public Linear(final Tensor weight, final Tensor bias) {
        // Pre-transpose and ensure native memory for zero-copy BLAS
        this.weightT = weight.transposeNative();
        this.bias = bias;
    }

    public Tensor forward(final Tensor x) {
        Tensor out = x.matmul(this.weightT);
        return this.bias != null ? out.addInPlace(this.bias) : out;
    }
}

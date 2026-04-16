package com.sparrowlogic.whisper4j.nn;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import org.jspecify.annotations.Nullable;
import java.util.Map;

/**
 * Residual attention block matching Whisper's architecture.
 * Uses in-place add for residual connections to reduce allocations.
 */
public final class ResidualAttentionBlock {

    private final MultiHeadAttention attn;
    private final LayerNorm attnLn;
    private final @Nullable MultiHeadAttention crossAttn;
    private final @Nullable LayerNorm crossAttnLn;
    private final Linear mlp1;
    private final Linear mlp2;
    private final LayerNorm mlpLn;

    public ResidualAttentionBlock(final MultiHeadAttention attn, final LayerNorm attnLn,
                                  final @Nullable MultiHeadAttention crossAttn,
                                  final @Nullable LayerNorm crossAttnLn,
                                  final Linear mlp1, final Linear mlp2, final LayerNorm mlpLn) {
        this.attn = attn;
        this.attnLn = attnLn;
        this.crossAttn = crossAttn;
        this.crossAttnLn = crossAttnLn;
        this.mlp1 = mlp1;
        this.mlp2 = mlp2;
        this.mlpLn = mlpLn;
    }

    public Tensor forward(final Tensor x, final @Nullable Tensor xa,
                          final @Nullable Tensor mask,
                          final @Nullable Map<String, Tensor> kvCache,
                          final String blockPrefix) {
        return this.forwardWithAttn(x, xa, mask, kvCache, blockPrefix)[0];
    }

    /** Forward pass returning [output, crossAttnWeights]. */
    public Tensor[] forwardWithAttn(final Tensor x, final @Nullable Tensor xa,
                                    final @Nullable Tensor mask,
                                    final @Nullable Map<String, Tensor> kvCache,
                                    final String blockPrefix) {
        Tensor[] attnOut = this.attn.forward(this.attnLn.forward(x), null, mask,
                kvCache, blockPrefix + ".attn");
        Tensor result = attnOut[0].addInPlace(x);

        Tensor crossAttnWeights = null;
        if (this.crossAttn != null) {
            Tensor[] crossOut = this.crossAttn.forward(
                    this.crossAttnLn.forward(result), xa, null, kvCache, blockPrefix + ".cross_attn");
            result = result.addInPlace(crossOut[0]);
            crossAttnWeights = crossOut[1];
        }

        Tensor mlpOut = this.mlp2.forward(this.mlp1.forward(this.mlpLn.forward(result)).geluInPlace());
        return new Tensor[]{result.addInPlace(mlpOut), crossAttnWeights};
    }
}

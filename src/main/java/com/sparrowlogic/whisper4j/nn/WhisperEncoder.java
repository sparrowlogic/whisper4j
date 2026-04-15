package com.sparrowlogic.whisper4j.nn;

import com.sparrowlogic.whisper4j.model.ModelDimensions;
import com.sparrowlogic.whisper4j.model.WeightStore;
import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.logging.Logger;

/**
 * Whisper audio encoder. Ported from openai/whisper (cba3768) whisper/model.py AudioEncoder.
 * conv1(n_mels, n_state, k=3, p=1) -&gt; GELU -&gt; conv2(n_state, n_state, k=3, s=2, p=1) -&gt; GELU
 * -&gt; + positional_embedding -&gt; N x ResidualAttentionBlock -&gt; ln_post
 *
 * Uses per-layer scoped arenas to batch-free intermediate tensors, avoiding
 * ~10GB of GC-triggered native memory churn across 32 encoder layers.
 */
public final class WhisperEncoder {

    private static final Logger LOG = Logger.getLogger(WhisperEncoder.class.getName());
    private final Conv1d conv1;
    private final Conv1d conv2;
    private final Tensor positionalEmbedding;
    private final ResidualAttentionBlock[] blocks;
    private final LayerNorm lnPost;

    public WhisperEncoder(final WeightStore w, final ModelDimensions dims) {
        String p = "encoder.";
        this.conv1 = new Conv1d(w.get(p + "conv1.weight"), w.get(p + "conv1.bias"), 1, 1);
        this.conv2 = new Conv1d(w.get(p + "conv2.weight"), w.get(p + "conv2.bias"), 2, 1);
        this.positionalEmbedding = w.get(p + "positional_embedding");
        this.lnPost = new LayerNorm(w.get(p + "ln_post.weight"), w.get(p + "ln_post.bias"));

        this.blocks = new ResidualAttentionBlock[dims.nAudioLayer()];
        for (int i = 0; i < dims.nAudioLayer(); i++) {
            String bp = p + "blocks." + i + ".";
            this.blocks[i] = buildEncoderBlock(w, bp, dims.nAudioHead());
        }
    }

    /** input: (batch, n_mels, n_ctx), output: (batch, n_audio_ctx, n_audio_state). */
    public Tensor forward(final Tensor mel) {
        LOG.fine("Encoder input: " + mel);
        long t0 = System.currentTimeMillis();
        // conv1 + GELU
        Tensor x = this.conv1.forward(mel).gelu();
        LOG.fine("Conv1+GELU: %d ms, output=%s".formatted(System.currentTimeMillis() - t0, x));
        t0 = System.currentTimeMillis();
        // conv2 + GELU (stride=2 halves the time dimension)
        x = this.conv2.forward(x).gelu();
        LOG.fine("Conv2+GELU: %d ms, output=%s".formatted(System.currentTimeMillis() - t0, x));
        // permute (batch, channels, time) -> (batch, time, channels)
        x = x.transpose();
        // add positional embedding
        x = x.add(this.positionalEmbedding);
        // encoder blocks — use per-layer scoped arena to batch-free intermediates
        for (int i = 0; i < this.blocks.length; i++) {
            long blockStart = System.currentTimeMillis();
            try (Arena layerArena = Arena.ofShared()) {
                Tensor.setScopedArena(layerArena);
                x = this.blocks[i].forward(x, null, null, null, "");
                // Copy output out of the scoped arena before it closes
                x = copyToAutoArena(x);
            } finally {
                Tensor.clearScopedArena();
            }
            LOG.fine("Encoder block %d/%d: %d ms, output=%s".formatted(
                    i + 1, this.blocks.length, System.currentTimeMillis() - blockStart, x));
        }
        Tensor out = this.lnPost.forward(x);
        // Debug: print first few values of encoder output
        float[] d = out.data();
        LOG.fine("Encoder output: " + out + " first8=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"
                .formatted(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]));
        return out;
    }

    /** Copy tensor data to a fresh Arena.ofAuto() allocation so it survives arena close. */
    private static Tensor copyToAutoArena(final Tensor src) {
        long bytes = (long) src.size() * Float.BYTES;
        MemorySegment dst = Arena.ofAuto().allocate(bytes, Float.BYTES);
        MemorySegment.copy(src.segment(), 0, dst, 0, bytes);
        return Tensor.ofSegment(dst, src.shape().clone());
    }

    private static ResidualAttentionBlock buildEncoderBlock(final WeightStore w, final String bp,
                                                            final int nHead) {
        return new ResidualAttentionBlock(
                buildMHA(w, bp + "attn.", nHead),
                new LayerNorm(w.get(bp + "attn_ln.weight"), w.get(bp + "attn_ln.bias")),
                null, null,
                new Linear(w.get(bp + "mlp.0.weight"), w.get(bp + "mlp.0.bias")),
                new Linear(w.get(bp + "mlp.2.weight"), w.get(bp + "mlp.2.bias")),
                new LayerNorm(w.get(bp + "mlp_ln.weight"), w.get(bp + "mlp_ln.bias"))
        );
    }

    public static MultiHeadAttention buildMHA(final WeightStore w, final String p, final int nHead) {
        return new MultiHeadAttention(nHead,
                new Linear(w.get(p + "query.weight"), w.get(p + "query.bias")),
                new Linear(w.get(p + "key.weight"), w.contains(p + "key.bias") ? w.get(p + "key.bias") : null),
                new Linear(w.get(p + "value.weight"), w.get(p + "value.bias")),
                new Linear(w.get(p + "out.weight"), w.get(p + "out.bias"))
        );
    }
}

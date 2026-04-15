package com.sparrowlogic.whisper4j.model;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ModelDimensionsTest {

    @Test
    void recordAccessors() {
        var dims = new ModelDimensions(80, 1500, 512, 8, 6, 51864, 448, 512, 8, 6);
        assertEquals(80, dims.nMels());
        assertEquals(1500, dims.nAudioCtx());
        assertEquals(512, dims.nAudioState());
        assertEquals(8, dims.nAudioHead());
        assertEquals(6, dims.nAudioLayer());
        assertEquals(51864, dims.nVocab());
        assertEquals(448, dims.nTextCtx());
        assertEquals(512, dims.nTextState());
        assertEquals(8, dims.nTextHead());
        assertEquals(6, dims.nTextLayer());
    }

    @Test
    void weightStoreBasicOps() {
        var store = new WeightStore();
        Tensor t = Tensor.of(new float[]{1f, 2f, 3f}, 3);
        store.put("test.weight", t);
        assertTrue(store.contains("test.weight"));
        assertFalse(store.contains("missing"));
        assertSame(t, store.get("test.weight"));
    }

    @Test
    void weightStoreGetMissingThrows() {
        var store = new WeightStore();
        assertThrows(IllegalArgumentException.class, () -> store.get("missing"));
    }

    @Test
    void weightStoreSize() {
        var store = new WeightStore();
        assertEquals(0, store.size());
        store.put("a", Tensor.of(new float[]{1f}, 1));
        store.put("b", Tensor.of(new float[]{2f}, 1));
        assertEquals(2, store.size());
    }

    @Test
    void weightStoreKeys() {
        var store = new WeightStore();
        store.put("x", Tensor.of(new float[]{1f}, 1));
        store.put("y", Tensor.of(new float[]{2f}, 1));
        var names = store.names();
        assertTrue(names.contains("x"));
        assertTrue(names.contains("y"));
        assertEquals(2, names.size());
    }

    @Test
    void weightStoreDimAndRank() {
        var store = new WeightStore();
        store.put("w", Tensor.of(new float[6], 2, 3));
        assertEquals(2, store.dim("w", 0));
        assertEquals(3, store.dim("w", 1));
        assertEquals(2, store.rank("w"));
    }

    @Test
    void normalizeKeyHuggingFace() {
        assertEquals("encoder.blocks.0.attn.query.weight",
                WeightStore.normalizeKey("model.encoder.layers.0.self_attn.q_proj.weight"));
        assertEquals("decoder.blocks.1.cross_attn.key.weight",
                WeightStore.normalizeKey("model.decoder.layers.1.encoder_attn.k_proj.weight"));
        assertEquals("encoder.ln_post.weight",
                WeightStore.normalizeKey("encoder.layer_norm.weight"));
        assertEquals("decoder.ln.bias",
                WeightStore.normalizeKey("decoder.layer_norm.bias"));
    }

    @Test
    void normalizeKeyOnnx() {
        assertEquals("encoder.conv1.weight",
                WeightStore.normalizeKey("/encoder/conv1/weight"));
    }

    @Test
    void normalizeKeyMlp() {
        assertEquals("encoder.blocks.0.mlp.0.weight",
                WeightStore.normalizeKey("model.encoder.layers.0.fc1.weight"));
        assertEquals("encoder.blocks.0.mlp.2.weight",
                WeightStore.normalizeKey("model.encoder.layers.0.fc2.weight"));
    }

    @Test
    void normalizeKeyLayerNorms() {
        assertEquals("encoder.blocks.0.attn_ln.weight",
                WeightStore.normalizeKey("model.encoder.layers.0.self_attn_layer_norm.weight"));
        assertEquals("decoder.blocks.0.cross_attn_ln.weight",
                WeightStore.normalizeKey("model.decoder.layers.0.encoder_attn_layer_norm.weight"));
        assertEquals("decoder.blocks.0.mlp_ln.weight",
                WeightStore.normalizeKey("model.decoder.layers.0.final_layer_norm.weight"));
    }

    @Test
    void normalizeNames() {
        var store = new WeightStore();
        store.put("model.encoder.layers.0.fc1.weight", Tensor.of(new float[]{1f}, 1));
        store.normalizeNames();
        assertTrue(store.contains("encoder.blocks.0.mlp.0.weight"));
        assertFalse(store.contains("model.encoder.layers.0.fc1.weight"));
    }

    @Test
    void normalizeKeyEmbeddings() {
        assertEquals("decoder.token_embedding.weight",
                WeightStore.normalizeKey("model.decoder.embed_tokens.weight"));
        assertEquals("encoder.positional_embedding",
                WeightStore.normalizeKey("model.encoder.embed_positions.weight"));
    }
}

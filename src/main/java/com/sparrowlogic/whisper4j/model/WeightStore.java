package com.sparrowlogic.whisper4j.model;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Stores named tensors loaded from a model file.
 * Common representation across all model formats.
 */
public final class WeightStore {

    private final Map<String, Tensor> weights = new LinkedHashMap<>();

    /** Store a named tensor. */
    public void put(final String name, final Tensor tensor) {
        this.weights.put(name, tensor);
    }

    /**
     * Retrieve a tensor by name.
     *
     * @param name canonical weight name (e.g. {@code "encoder.conv1.weight"})
     * @return the tensor
     * @throws IllegalArgumentException if the name is not found
     */
    public Tensor get(final String name) {
        Tensor t = this.weights.get(name);
        if (t == null) {
            throw new IllegalArgumentException("weight not found: " + name);
        }
        return t;
    }

    /** Check if a weight with the given name exists. */
    public boolean contains(final String name) {
        return this.weights.containsKey(name);
    }

    public int dim(final String name, final int axis) {
        return this.get(name).dim(axis);
    }

    public int rank(final String name) {
        return this.get(name).rank();
    }

    public Set<String> names() {
        return this.weights.keySet();
    }

    public int size() {
        return this.weights.size();
    }

    /**
     * Normalize weight names from format-specific conventions to canonical Whisper names.
     * Canonical names match GGML/whisper.cpp: encoder.conv1.weight, decoder.blocks.0.attn.query.weight
     *
     * <p>Handles:
     * - HuggingFace SafeTensors: model. prefix + different naming (layers→blocks, self_attn→attn, etc.)
     * - ONNX: replaces "/" with ".", strips leading "."
     * - CTranslate2: already handled by CT2Loader (slash→dot)
     * - PyTorch: already canonical (from state_dict)
     */
    public void normalizeNames() {
        Map<String, Tensor> normalized = new LinkedHashMap<>();
        for (final var entry : this.weights.entrySet()) {
            String name = normalizeKey(entry.getKey());
            normalized.put(name, entry.getValue());
        }
        this.weights.clear();
        this.weights.putAll(normalized);
    }

    @SuppressWarnings("checkstyle:CyclomaticComplexity")
    public static String normalizeKey(final String key) {
        String name = key;
        // ONNX: /encoder/conv1/weight → encoder.conv1.weight
        name = name.replace('/', '.');
        if (name.startsWith(".")) {
            name = name.substring(1);
        }
        // HuggingFace: model.encoder... → encoder...
        if (name.startsWith("model.")) {
            name = name.substring(6);
        }

        // HuggingFace → GGML name mapping
        // Embeddings
        name = name.replace("embed_tokens.weight", "token_embedding.weight");
        name = name.replace("embed_positions.weight", "positional_embedding");
        // Blocks
        name = name.replace(".layers.", ".blocks.");
        // Attention projections
        name = name.replace(".self_attn.q_proj.", ".attn.query.");
        name = name.replace(".self_attn.k_proj.", ".attn.key.");
        name = name.replace(".self_attn.v_proj.", ".attn.value.");
        name = name.replace(".self_attn.out_proj.", ".attn.out.");
        name = name.replace(".encoder_attn.q_proj.", ".cross_attn.query.");
        name = name.replace(".encoder_attn.k_proj.", ".cross_attn.key.");
        name = name.replace(".encoder_attn.v_proj.", ".cross_attn.value.");
        name = name.replace(".encoder_attn.out_proj.", ".cross_attn.out.");
        // Layer norms
        name = name.replace(".self_attn_layer_norm.", ".attn_ln.");
        name = name.replace(".encoder_attn_layer_norm.", ".cross_attn_ln.");
        name = name.replace(".final_layer_norm.", ".mlp_ln.");
        // MLP
        name = name.replace(".fc1.", ".mlp.0.");
        name = name.replace(".fc2.", ".mlp.2.");
        // Encoder/decoder final layer norm
        if ("encoder.layer_norm.weight".equals(name) || "encoder.layer_norm.bias".equals(name)) {
            name = name.replace("encoder.layer_norm.", "encoder.ln_post.");
        }
        if ("decoder.layer_norm.weight".equals(name) || "decoder.layer_norm.bias".equals(name)) {
            name = name.replace("decoder.layer_norm.", "decoder.ln.");
        }
        return name;
    }
}

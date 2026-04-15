package com.sparrowlogic.whisper4j.model;

/**
 * Whisper model architecture dimensions, matching OpenAI's ModelDimensions dataclass.
 */
public record ModelDimensions(
        int nMels,
        int nAudioCtx,
        int nAudioState,
        int nAudioHead,
        int nAudioLayer,
        int nVocab,
        int nTextCtx,
        int nTextState,
        int nTextHead,
        int nTextLayer
) {

    /** Infer dimensions from weight shapes in a WeightStore. */
    public static ModelDimensions infer(final WeightStore weights) {
        int nAudioState = weights.dim("encoder.conv1.weight", 0);
        int nMels = weights.dim("encoder.conv1.weight", 1);
        // count encoder blocks
        int nAudioLayer = 0;
        while (weights.contains("encoder.blocks." + nAudioLayer + ".attn.query.weight")) {
            nAudioLayer++;
        }
        // count decoder blocks
        int nTextLayer = 0;
        while (weights.contains("decoder.blocks." + nTextLayer + ".attn.query.weight")) {
            nTextLayer++;
        }

        int nVocab = weights.dim("decoder.token_embedding.weight", 0);
        int nTextState = weights.dim("decoder.token_embedding.weight", 1);
        int nTextCtx = weights.dim("decoder.positional_embedding", 0);
        int nAudioCtx = weights.dim("encoder.positional_embedding", 0);

        // infer head count from query weight shape: Whisper uses headDim = 64 for all models
        int nAudioHead = nAudioState / 64;
        int nTextHead = nTextState / 64;

        return new ModelDimensions(nMels, nAudioCtx, nAudioState, nAudioHead, nAudioLayer,
                nVocab, nTextCtx, nTextState, nTextHead, nTextLayer);
    }
}

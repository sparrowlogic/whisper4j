package com.sparrowlogic.whisper4j.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Loads model weights from a specific format into a WeightStore.
 */
public sealed interface ModelLoader
        permits SafeTensorsLoader, PyTorchLoader, CTranslate2Loader, OnnxLoader, GgmlLoader {

    /**
     * Load model weights from the given path into a {@link WeightStore}.
     *
     * @param path path to the model file
     * @return weight store containing all named tensors
     * @throws IOException if the file cannot be read or is corrupt
     */
    WeightStore load(Path path) throws IOException;

    /**
     * Auto-detect model format from file extension or directory contents.
     *
     * @param path path to model file or HuggingFace directory
     * @return appropriate loader for the detected format
     */
    @SuppressWarnings("checkstyle:ReturnCount")
    static ModelLoader forPath(final Path path) {
        String name = path.getFileName().toString().toLowerCase();
        if (name.endsWith(".safetensors")) {
            return new SafeTensorsLoader();
        }
        if (name.endsWith(".pt") || name.equals("pytorch_model.bin")) {
            return new PyTorchLoader();
        }
        if (name.endsWith(".onnx")) {
            return new OnnxLoader();
        }
        if ((name.startsWith("ggml") || name.startsWith("for-tests-ggml")) && name.endsWith(".bin")) {
            return new GgmlLoader();
        }
        // Directory: check for split ONNX (encoder_model.onnx + decoder_model.onnx)
        if (path.toFile().isDirectory()) {
            if (path.resolve("encoder_model.onnx").toFile().exists()
                    || path.resolve("onnx/encoder_model.onnx").toFile().exists()) {
                return new OnnxLoader();
            }
            if (path.resolve("model.bin").toFile().exists()
                    && path.resolve("config.json").toFile().exists()) {
                return new CTranslate2Loader();
            }
        }
        // fallback: try to detect by reading magic bytes
        return new GgmlLoader();
    }
}

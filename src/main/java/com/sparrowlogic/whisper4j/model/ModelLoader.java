package com.sparrowlogic.whisper4j.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Loads model weights from a specific format into a WeightStore.
 */
public sealed interface ModelLoader
        permits SafeTensorsLoader, PyTorchLoader, CTranslate2Loader, OnnxLoader, GgmlLoader {

    WeightStore load(Path path) throws IOException;

    /** Auto-detect format from file/directory and return appropriate loader. */
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

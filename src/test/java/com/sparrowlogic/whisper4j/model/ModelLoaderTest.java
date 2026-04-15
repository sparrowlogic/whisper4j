package com.sparrowlogic.whisper4j.model;

import org.junit.jupiter.api.Test;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class ModelLoaderTest {

    @Test
    void detectGgml() {
        var loader = ModelLoader.forPath(Path.of("ggml-base.en.bin"));
        assertInstanceOf(GgmlLoader.class, loader);
    }

    @Test
    void detectSafeTensors() {
        var loader = ModelLoader.forPath(Path.of("model.safetensors"));
        assertInstanceOf(SafeTensorsLoader.class, loader);
    }

    @Test
    void detectPyTorchPt() {
        var loader = ModelLoader.forPath(Path.of("model.pt"));
        assertInstanceOf(PyTorchLoader.class, loader);
    }

    @Test
    void detectPyTorchBin() {
        var loader = ModelLoader.forPath(Path.of("pytorch_model.bin"));
        assertInstanceOf(PyTorchLoader.class, loader);
    }

    @Test
    void detectOnnx() {
        var loader = ModelLoader.forPath(Path.of("model.onnx"));
        assertInstanceOf(OnnxLoader.class, loader);
    }

    @Test
    void detectTestGgml() {
        var loader = ModelLoader.forPath(Path.of("for-tests-ggml-base.en.bin"));
        assertInstanceOf(GgmlLoader.class, loader);
    }

    @Test
    void unknownFallsBackToGgml() {
        var loader = ModelLoader.forPath(Path.of("unknown.dat"));
        assertInstanceOf(GgmlLoader.class, loader);
    }
}

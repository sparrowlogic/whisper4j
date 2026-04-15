package com.sparrowlogic.whisper4j.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class GgmlLoaderTest {

    static final Path TINY_EN = Path.of("models/ggml-tiny.en.bin");
    static final Path BASE_EN = Path.of("models/ggml-base.en.bin");

    static boolean tinyEnExists() { return Files.exists(TINY_EN); }
    static boolean baseEnExists() { return Files.exists(BASE_EN); }

    @Test
    @EnabledIf("tinyEnExists")
    void loadTinyEnModel() throws IOException {
        var loader = new GgmlLoader();
        WeightStore weights = loader.load(TINY_EN);
        ModelDimensions dims = loader.dimensions();

        System.out.println("Loaded tiny.en: " + weights.size() + " tensors");
        System.out.println("Dims: " + dims);
        System.out.println("Vocab size: " + loader.vocab().size());

        // tiny model: 4 encoder layers, 4 decoder layers, 384 state, 6 heads
        assertEquals(4, dims.nAudioLayer());
        assertEquals(4, dims.nTextLayer());
        assertEquals(384, dims.nAudioState());
        assertEquals(6, dims.nAudioHead());
        assertEquals(80, dims.nMels());
        assertEquals(51864, dims.nVocab()); // English-only

        // verify key tensors exist
        assertTrue(weights.contains("encoder.conv1.weight"));
        assertTrue(weights.contains("encoder.conv1.bias"));
        assertTrue(weights.contains("encoder.blocks.0.attn.query.weight"));
        assertTrue(weights.contains("decoder.token_embedding.weight"));
        assertTrue(weights.contains("decoder.blocks.0.cross_attn.query.weight"));

        // verify vocab has tokens
        assertTrue(loader.vocab().size() > 50000);
        assertNotNull(loader.vocab().get(0));
    }

    @Test
    @EnabledIf("baseEnExists")
    void loadBaseEnModel() throws IOException {
        var loader = new GgmlLoader();
        WeightStore weights = loader.load(BASE_EN);
        ModelDimensions dims = loader.dimensions();

        System.out.println("Loaded base.en: " + weights.size() + " tensors");
        System.out.println("Dims: " + dims);

        // base model: 6 encoder layers, 6 decoder layers, 512 state, 8 heads
        assertEquals(6, dims.nAudioLayer());
        assertEquals(6, dims.nTextLayer());
        assertEquals(512, dims.nAudioState());
        assertEquals(8, dims.nAudioHead());
    }
}

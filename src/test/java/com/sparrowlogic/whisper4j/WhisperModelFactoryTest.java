package com.sparrowlogic.whisper4j;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WhisperModelFactoryTest {

    @AfterEach
    void cleanup() {
        WhisperModelFactory.clearRegistry();
    }

    // ---- Builder tests ----

    @Test
    void buildOptionsReflectsSettings() {
        var opts = new WhisperModelFactory(Path.of("model.bin"))
                .language("de")
                .task("translate")
                .beamSize(3)
                .withTimestamps(false)
                .vadFilter(true)
                .conditionOnPrevious(true)
                .buildOptions();

        assertEquals("de", opts.language());
        assertEquals("translate", opts.task());
        assertEquals(3, opts.beamSize());
        assertFalse(opts.withTimestamps());
        assertTrue(opts.vadFilter());
        assertTrue(opts.conditionOnPrevious());
    }

    @Test
    void buildOptionsDefaults() {
        var opts = new WhisperModelFactory(Path.of("model.bin")).buildOptions();
        assertEquals("en", opts.language());
        assertEquals("transcribe", opts.task());
        assertEquals(5, opts.beamSize());
        assertTrue(opts.withTimestamps());
    }

    @Test
    void createWithInvalidPathThrows() {
        var factory = new WhisperModelFactory(Path.of("nonexistent.bin"));
        assertThrows(IOException.class, factory::create);
    }

    @Test
    void createLoadsModel() throws IOException {
        Path modelPath = Path.of("models/ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "Model not found: " + modelPath);
        WhisperModel model = new WhisperModelFactory(modelPath).beamSize(1).create();
        assertNotNull(model);
        model.close();
    }

    // ---- Static registry tests ----

    @Test
    void sharedUnregisteredThrows() {
        assertThrows(IllegalArgumentException.class, () -> WhisperModelFactory.shared("missing"));
    }

    @Test
    void sharedWithBadPathThrowsUncheckedIO() {
        WhisperModelFactory.register("bad", new WhisperModelFactory(Path.of("nonexistent.bin")));
        assertThrows(UncheckedIOException.class, () -> WhisperModelFactory.shared("bad"));
    }

    @Test
    void isRegisteredReflectsState() {
        assertFalse(WhisperModelFactory.isRegistered("test"));
        WhisperModelFactory.register("test", new WhisperModelFactory(Path.of("model.bin")));
        assertTrue(WhisperModelFactory.isRegistered("test"));
    }

    @Test
    void sharedReturnsSameInstance() {
        Path modelPath = Path.of("models/ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "Model not found: " + modelPath);

        WhisperModelFactory.register("tiny", new WhisperModelFactory(modelPath).beamSize(1));
        WhisperModel first = WhisperModelFactory.shared("tiny");
        WhisperModel second = WhisperModelFactory.shared("tiny");
        assertSame(first, second);
    }

    // ---- Eviction tests ----

    @Test
    void evictRemovesLoadedModel() {
        Path modelPath = Path.of("models/ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "Model not found: " + modelPath);

        WhisperModelFactory.register("tiny", new WhisperModelFactory(modelPath).beamSize(1));
        WhisperModel before = WhisperModelFactory.shared("tiny");
        assertEquals(1, WhisperModelFactory.loadedCount());

        WhisperModelFactory.evict("tiny");
        assertEquals(0, WhisperModelFactory.loadedCount());

        // Re-access reloads a new instance
        WhisperModel after = WhisperModelFactory.shared("tiny");
        assertNotSame(before, after);
    }

    @Test
    void evictLruKeepsRecentlyUsed() {
        Path tinyPath = Path.of("models/ggml-tiny.en.bin");
        Path basePath = Path.of("models/ggml-base.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(tinyPath) && Files.exists(basePath), "Models not found");

        WhisperModelFactory.register("a", new WhisperModelFactory(tinyPath).beamSize(1));
        WhisperModelFactory.register("b", new WhisperModelFactory(basePath).beamSize(1));

        // Load both — "a" first, then "b"
        WhisperModelFactory.shared("a");
        WhisperModelFactory.shared("b");
        assertEquals(2, WhisperModelFactory.loadedCount());

        // Touch "a" again to make it more recent
        WhisperModelFactory.shared("a");

        // Evict down to 1 — should keep "a" (most recent), evict "b" (least recent)
        int evicted = WhisperModelFactory.evictLeastRecentlyUsed(1);
        assertEquals(1, evicted);
        assertEquals(1, WhisperModelFactory.loadedCount());

        // "a" should still be loaded (same instance)
        assertNotNull(WhisperModelFactory.shared("a"));
    }

    @Test
    void evictLruNoOpWhenUnderLimit() {
        assertEquals(0, WhisperModelFactory.evictLeastRecentlyUsed(5));
    }

    @Test
    void loadedCountTracksState() {
        assertEquals(0, WhisperModelFactory.loadedCount());
        Path modelPath = Path.of("models/ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "Model not found: " + modelPath);

        WhisperModelFactory.register("x", new WhisperModelFactory(modelPath).beamSize(1));
        assertEquals(0, WhisperModelFactory.loadedCount()); // registered but not loaded
        WhisperModelFactory.shared("x");
        assertEquals(1, WhisperModelFactory.loadedCount());
    }
}

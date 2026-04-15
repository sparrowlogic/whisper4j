package com.sparrowlogic.whisper4j;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Validates that WhisperModel is thread-safe: load once, transcribe concurrently.
 * All model weights are immutable (final fields). KV caches and intermediate tensors
 * are allocated per-call. The scoped arena in Tensor is ThreadLocal.
 */
class ConcurrencyTest {

    private static final Logger LOG = Logger.getLogger(ConcurrencyTest.class.getName());
    private static final String MODELS_DIR = "models/";
    private static final String AUDIO = "src/test/resources/data/physicsworks.wav";

    @Test
    void concurrentTranscribeProducesSameResults() throws Exception {
        Path modelPath = Path.of(MODELS_DIR + "ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "tiny.en not found");

        WhisperModel model = WhisperModel.load(modelPath);
        var wav = com.sparrowlogic.whisper4j.audio.WavParser.parse(Path.of(AUDIO));
        float[] mono = com.sparrowlogic.whisper4j.audio.Resampler.toWhisperInput(
                wav.samples(), wav.sampleRate(), wav.channels());

        // Take 3 different 10s clips
        float[][] clips = new float[3][16000 * 10];
        for (int i = 0; i < 3; i++) {
            int offset = i * 16000 * 30;
            System.arraycopy(mono, offset, clips[i], 0, clips[i].length);
        }

        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", 1, false, false, false);

        // First: get serial baseline results
        String[] baseline = new String[3];
        for (int i = 0; i < 3; i++) {
            StringBuilder sb = new StringBuilder();
            model.transcribe(clips[i], opts).forEach(s -> sb.append(s.text()));
            baseline[i] = sb.toString();
        }

        // Now: run all 3 concurrently
        int threads = 3;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch startGate = new CountDownLatch(1);
        ConcurrentHashMap<Integer, String> results = new ConcurrentHashMap<>();
        List<Future<?>> futures = new ArrayList<>();

        for (int i = 0; i < threads; i++) {
            final int idx = i;
            futures.add(pool.submit(() -> {
                try {
                    startGate.await(); // all threads start together
                    StringBuilder sb = new StringBuilder();
                    model.transcribe(clips[idx], opts).forEach(s -> sb.append(s.text()));
                    results.put(idx, sb.toString());
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }));
        }

        startGate.countDown(); // release all threads
        for (Future<?> f : futures) {
            f.get(120, TimeUnit.SECONDS);
        }
        pool.shutdown();

        // Verify: concurrent results must match serial baseline
        for (int i = 0; i < threads; i++) {
            assertEquals(baseline[i], results.get(i),
                    "Concurrent result for clip " + i + " should match serial baseline");
        }
        LOG.info("Concurrency test passed: 3 threads, all results match serial baseline");
    }

    @Test
    void concurrentTranscribeSameClip() throws Exception {
        Path modelPath = Path.of(MODELS_DIR + "ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "tiny.en not found");

        WhisperModel model = WhisperModel.load(modelPath);
        float[] clip = new float[16000 * 5];
        // Simple sine wave — deterministic input
        for (int i = 0; i < clip.length; i++) {
            clip[i] = 0.3f * (float) Math.sin(2 * Math.PI * 440 * i / 16000.0);
        }

        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", 1, false, false, false);

        // Get baseline
        StringBuilder baselineSb = new StringBuilder();
        model.transcribe(clip, opts).forEach(s -> baselineSb.append(s.text()));
        String baseline = baselineSb.toString();

        // Run 4 threads on the same clip
        int threads = 4;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<String>> futures = new ArrayList<>();

        for (int i = 0; i < threads; i++) {
            futures.add(pool.submit(() -> {
                StringBuilder sb = new StringBuilder();
                model.transcribe(clip, opts).forEach(s -> sb.append(s.text()));
                return sb.toString();
            }));
        }

        for (Future<String> f : futures) {
            String result = f.get(60, TimeUnit.SECONDS);
            assertEquals(baseline, result, "All threads should produce identical output");
        }
        pool.shutdown();
        LOG.info("Same-clip concurrency test passed: 4 threads, identical output");
    }

    @Test
    void closeBlocksUntilTranscriptionCompletes() throws Exception {
        Path modelPath = Path.of(MODELS_DIR + "ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "Model not found: " + modelPath);

        WhisperModel model = WhisperModel.load(modelPath);
        var wav = com.sparrowlogic.whisper4j.audio.WavParser.parse(Path.of(AUDIO));
        float[] mono = com.sparrowlogic.whisper4j.audio.Resampler.toWhisperInput(
                wav.samples(), wav.sampleRate(), wav.channels());
        float[] clip = new float[16000 * 10];
        System.arraycopy(mono, 0, clip, 0, clip.length);
        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", 1, false, false, false);

        // Start transcription on background thread
        CountDownLatch started = new CountDownLatch(1);
        var future = Executors.newSingleThreadExecutor().submit(() -> {
            started.countDown();
            return model.transcribe(clip, opts).toList();
        });

        started.await(); // wait for transcription to begin
        // close() should block until transcription finishes, not crash it
        model.close();
        // The transcription should have completed successfully
        var segments = future.get(30, TimeUnit.SECONDS);
        assertNotNull(segments);
        LOG.info("Close-during-transcription test passed");
    }

    @Test
    void transcribeAfterCloseThrows() throws Exception {
        Path modelPath = Path.of(MODELS_DIR + "ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "Model not found: " + modelPath);

        WhisperModel model = WhisperModel.load(modelPath);
        model.close();
        assertThrows(IllegalStateException.class,
                () -> model.transcribe(new float[16000], opts()));
    }

    @Test
    void cancelStopsTranscriptionEarly() throws Exception {
        Path modelPath = Path.of(MODELS_DIR + "ggml-tiny.en.bin");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(modelPath), "Model not found: " + modelPath);

        WhisperModel model = WhisperModel.load(modelPath);
        var wav = com.sparrowlogic.whisper4j.audio.WavParser.parse(Path.of(AUDIO));
        float[] mono = com.sparrowlogic.whisper4j.audio.Resampler.toWhisperInput(
                wav.samples(), wav.sampleRate(), wav.channels());

        // Get full result for comparison
        var fullSegments = model.transcribe(mono, opts()).toList();

        // Cancel after 100ms — should get partial results
        var handle = new WhisperModel.TranscriptionHandle();
        var future = Executors.newSingleThreadExecutor().submit(() ->
                model.transcribe(mono, opts(), handle).toList());

        Thread.sleep(100);
        handle.cancel();

        var partial = future.get(30, TimeUnit.SECONDS);
        assertTrue(partial.size() < fullSegments.size(),
                "Cancelled should have fewer segments: partial=" + partial.size()
                        + " full=" + fullSegments.size());
        assertTrue(handle.isCancelled());
        LOG.info("Cancel test: %d partial vs %d full segments".formatted(
                partial.size(), fullSegments.size()));
        model.close();
    }

    private static WhisperModel.TranscriptionOptions opts() {
        return new WhisperModel.TranscriptionOptions("en", "transcribe", 1, false, false, false);
    }
}

package com.sparrowlogic.whisper4j;

import javax.sound.sampled.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Live streaming microphone transcription.
 *
 * Architecture:
 *   [Mic Thread] → captures 5s chunks → [Queue] → [Transcription Thread] → prints results
 *
 * The mic thread never blocks on transcription. If transcription falls behind,
 * chunks are queued (up to 3 deep) and a warning is printed. Audio capture
 * continues uninterrupted regardless of transcription speed.
 */
public class TestMicrophone {

    private static final int SAMPLE_RATE = 16000;
    private static final int CHUNK_SECONDS = 5;
    private static final int CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS;
    private static final int CHUNK_BYTES = CHUNK_SAMPLES * 2; // 16-bit mono
    private static final float SILENCE_THRESHOLD = 0.01f;

    public static void main(String[] args) throws Exception {
        configureLogging();

        String modelPath = args.length > 0 ? args[0]
                : "models/ggml-large-v3-turbo.bin";
        int beamSize = args.length > 1 ? Integer.parseInt(args[1]) : 1;
        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", beamSize, true, false, false);

        System.out.println("Loading model: " + modelPath);
        long loadStart = System.currentTimeMillis();
        WhisperModel model = WhisperModel.load(Path.of(modelPath));
        System.out.printf("Model loaded in %d ms: %s%n", System.currentTimeMillis() - loadStart, model.dimensions());

        // Warmup: run transcriptions on silence to JIT-compile hot paths
        System.out.println("Warming up JIT (3 iterations)...");
        long warmStart = System.currentTimeMillis();
        for (int i = 0; i < 3; i++) {
            long t = System.currentTimeMillis();
            model.transcribe(new float[CHUNK_SAMPLES], opts).forEach(_ -> {});
            System.out.printf("  warmup %d: %d ms%n", i + 1, System.currentTimeMillis() - t);
        }
        System.out.printf("Warmup done in %d ms%n%n", System.currentTimeMillis() - warmStart);

        // Audio chunk queue: mic thread produces, transcription thread consumes
        BlockingQueue<TimestampedChunk> queue = new ArrayBlockingQueue<>(3);
        AtomicBoolean running = new AtomicBoolean(true);
        AtomicInteger chunkNum = new AtomicInteger(0);
        AtomicInteger transcribedNum = new AtomicInteger(0);

        // Open mic — list available devices and select MacBook built-in mic
        AudioFormat format = new AudioFormat(SAMPLE_RATE, 16, 1, true, false);
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

        System.out.println("Available audio input devices:");
        Mixer.Info[] mixers = AudioSystem.getMixerInfo();
        Mixer.Info selectedMixer = null;
        for (Mixer.Info mi : mixers) {
            Mixer mixer = AudioSystem.getMixer(mi);
            if (mixer.getTargetLineInfo().length > 0) {
                boolean supported = false;
                for (Line.Info li : mixer.getTargetLineInfo()) {
                    if (li instanceof DataLine.Info dli && dli.isFormatSupported(format)) {
                        supported = true;
                    }
                }
                if (supported) {
                    String label = mi.getName() + " — " + mi.getDescription();
                    boolean isBuiltIn = mi.getName().toLowerCase().contains("macbook")
                            || mi.getName().toLowerCase().contains("built-in");
                    System.out.println("  " + (isBuiltIn ? "→ " : "  ") + label);
                    if (isBuiltIn || selectedMixer == null) {
                        selectedMixer = mi;
                    }
                }
            }
        }

        TargetDataLine mic;
        if (selectedMixer != null) {
            System.out.println("Using: " + selectedMixer.getName());
            Mixer mixer = AudioSystem.getMixer(selectedMixer);
            mic = (TargetDataLine) mixer.getLine(info);
        } else {
            System.out.println("Using: system default");
            mic = (TargetDataLine) AudioSystem.getLine(info);
        }
        mic.open(format);
        mic.start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            running.set(false);
            mic.stop();
            mic.close();
            System.out.println("\nShutdown. Transcribed " + transcribedNum.get() + " chunks.");
        }));

        // Mic capture thread — never blocks on transcription
        Thread micThread = Thread.ofPlatform().name("mic-capture").start(() -> {
            byte[] buffer = new byte[CHUNK_BYTES];
            while (running.get()) {
                int bytesRead = 0;
                while (bytesRead < CHUNK_BYTES && running.get()) {
                    int n = mic.read(buffer, bytesRead, CHUNK_BYTES - bytesRead);
                    if (n < 0) { break; }
                    bytesRead += n;
                }
                if (!running.get()) { break; }

                int num = chunkNum.incrementAndGet();
                float[] pcm = pcmBytesToFloat(buffer, bytesRead);
                float rms = rms(pcm);

                if (rms < SILENCE_THRESHOLD) {
                    System.out.printf("[chunk %d] (silence, rms=%.4f)%n", num, rms);
                    continue;
                }

                var chunk = new TimestampedChunk(num, pcm, rms, System.currentTimeMillis());
                if (!queue.offer(chunk)) {
                    System.out.printf("[chunk %d] ⚠ Queue full — transcription falling behind, dropping chunk%n", num);
                }
            }
        });

        // Transcription thread — processes chunks as fast as possible
        System.out.println("Ready to transcribe.");
        System.out.println("=== Listening (Ctrl+C to stop) ===");
        System.out.printf("Model: %s | Chunk: %ds | Beam: %d | Queue depth: %d%n%n",
                Path.of(modelPath).getFileName(), CHUNK_SECONDS, beamSize,
                queue.remainingCapacity() + queue.size());

        while (running.get()) {
            TimestampedChunk chunk;
            try {
                chunk = queue.take();
            } catch (InterruptedException e) {
                break;
            }

            long latency = System.currentTimeMillis() - chunk.capturedAt;
            int pending = queue.size();
            System.out.printf("[chunk %d] Transcribing (rms=%.4f, queue=%d, latency=%dms)...%n",
                    chunk.num, chunk.rms, pending, latency);

            long t0 = System.currentTimeMillis();
            StringBuilder result = new StringBuilder();
            model.transcribe(chunk.pcm, opts).forEach(seg -> {
                if (!seg.text().isBlank()) {
                    result.append(seg.text().strip());
                }
            });

            long elapsed = System.currentTimeMillis() - t0;
            transcribedNum.incrementAndGet();

            if (result.isEmpty()) {
                System.out.printf("[chunk %d] (no speech) %dms%n%n", chunk.num, elapsed);
            } else {
                double rtf = CHUNK_SECONDS * 1000.0 / elapsed;
                System.out.printf("[chunk %d] \"%s\" — %dms (%.1fx real-time)%n%n",
                        chunk.num, result, elapsed, rtf);
            }
        }

        micThread.join(1000);
    }

    record TimestampedChunk(int num, float[] pcm, float rms, long capturedAt) { }

    private static float[] pcmBytesToFloat(byte[] pcm, int length) {
        int samples = length / 2;
        float[] out = new float[samples];
        ByteBuffer bb = ByteBuffer.wrap(pcm, 0, length).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < samples; i++) {
            out[i] = bb.getShort() / 32768.0f;
        }
        return out;
    }

    private static float rms(float[] samples) {
        double sum = 0;
        for (float s : samples) { sum += s * s; }
        return (float) Math.sqrt(sum / samples.length);
    }

    private static void configureLogging() {
        // Only show INFO and above to keep output clean during streaming
        Logger root = Logger.getLogger("com.sparrowlogic.whisper4j");
        root.setLevel(Level.INFO);
        root.setUseParentHandlers(false);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.INFO);
        root.addHandler(handler);
    }
}

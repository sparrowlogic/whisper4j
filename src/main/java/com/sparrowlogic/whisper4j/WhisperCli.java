package com.sparrowlogic.whisper4j;

import com.sparrowlogic.whisper4j.audio.Resampler;
import com.sparrowlogic.whisper4j.audio.WavParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

/**
 * CLI entry point for transcription and benchmarking.
 * Usage: whisper4j &lt;model&gt; &lt;audio.wav&gt; [beam_size]
 */
public final class WhisperCli {

    private WhisperCli() { }

    public static void main(final String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: whisper4j <model> <audio.wav> [beam_size]");
            System.exit(1);
        }
        final Path modelPath = Path.of(args[0]);
        final Path audioPath = Path.of(args[1]);
        final int beam = args.length > 2 ? Integer.parseInt(args[2]) : 1;

        if (!Files.exists(modelPath)) {
            System.err.println("Model not found: " + modelPath);
            System.exit(1);
        }
        if (!Files.exists(audioPath)) {
            System.err.println("Audio not found: " + audioPath);
            System.exit(1);
        }

        final var opts = new WhisperModel.TranscriptionOptions(
                "en", "transcribe", beam, false, false, false);

        final long loadStart = System.currentTimeMillis();
        final WhisperModel model = WhisperModel.load(modelPath);
        final long loadMs = System.currentTimeMillis() - loadStart;

        final var wav = WavParser.parse(audioPath);
        final float[] mono = Resampler.toWhisperInput(
                wav.samples(), wav.sampleRate(), wav.channels());
        final float duration = mono.length / 16000.0f;

        final long t0 = System.currentTimeMillis();
        final List<WhisperModel.Segment> segments = model.transcribe(mono, opts).toList();
        final long totalMs = System.currentTimeMillis() - t0;
        final float rtf = duration * 1000.0f / totalMs;

        final String text = segments.stream()
                .map(WhisperModel.Segment::text)
                .collect(Collectors.joining(" ")).trim();

        System.out.printf("{\"model\":\"%s\",\"load_ms\":%d,\"total_ms\":%d,"
                        + "\"rtf\":%.1f,\"duration_s\":%.1f,\"segments\":%d,"
                        + "\"java\":\"%s\",\"text\":\"%s\"}%n",
                modelPath.getFileName(), loadMs, totalMs, rtf, duration,
                segments.size(), System.getProperty("java.version"),
                text.replace("\"", "\\\""));
    }
}

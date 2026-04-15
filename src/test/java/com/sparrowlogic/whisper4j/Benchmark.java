package com.sparrowlogic.whisper4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Benchmark all available Whisper models against physicsworks.wav (203s).
 * Outputs parseable JSON with timing and transcript text.
 * Usage: Benchmark [beam_size]
 *        -Dmodel=turbo  to filter to a single model
 */
public class Benchmark {

    static final String MODELS_DIR = "models/";
    static final String AUDIO = "src/test/resources/data/physicsworks.wav";
    static final String[] MODEL_FILES = {
            "ggml-tiny.en.bin",
            "ggml-base.en.bin",
            "ggml-small.en.bin",
            "ggml-medium.en.bin",
            "ggml-large-v3-turbo.bin",
            "ggml-large-v3.bin",
    };

    public static void main(String[] args) throws Exception {
        int beam = args.length > 0 ? Integer.parseInt(args[0]) : 1;
        String filter = System.getProperty("model", "");
        var opts = new WhisperModel.TranscriptionOptions("en", "transcribe", beam, false, false, false);
        var wav = com.sparrowlogic.whisper4j.audio.WavParser.parse(Path.of(AUDIO));
        float[] mono = com.sparrowlogic.whisper4j.audio.Resampler.toWhisperInput(
                wav.samples(), wav.sampleRate(), wav.channels());
        float duration = mono.length / 16000.0f;

        var sb = new StringBuilder();
        sb.append("{\"runtime\":\"whisper4j\",\"java_version\":\"")
                .append(System.getProperty("java.version"))
                .append("\",\"audio\":\"").append(AUDIO)
                .append("\",\"audio_duration_s\":").append(String.format("%.1f", duration))
                .append(",\"beam_size\":").append(beam)
                .append(",\"models\":[");

        boolean first = true;
        for (String modelFile : MODEL_FILES) {
            Path modelPath = Path.of(MODELS_DIR + modelFile);
            if (!Files.exists(modelPath)) { continue; }
            if (!filter.isEmpty() && !modelFile.contains(filter)) { continue; }

            if (!first) { sb.append(","); }
            first = false;

            try {
                long loadStart = System.currentTimeMillis();
                WhisperModel model = WhisperModel.load(modelPath);
                long loadMs = System.currentTimeMillis() - loadStart;

                model.transcribe(new float[16000 * 5], opts).forEach(s -> { });

                long t0 = System.currentTimeMillis();
                List<WhisperModel.Segment> segments = model.transcribe(mono, opts).toList();
                long totalMs = System.currentTimeMillis() - t0;
                float rtf = duration * 1000.0f / totalMs;

                String text = segments.stream().map(WhisperModel.Segment::text)
                        .collect(Collectors.joining(" ")).trim();

                sb.append("{\"model\":\"").append(modelFile).append("\"")
                        .append(",\"load_ms\":").append(loadMs)
                        .append(",\"total_ms\":").append(totalMs)
                        .append(",\"rtf\":").append(String.format("%.1f", rtf))
                        .append(",\"segments\":").append(segments.size())
                        .append(",\"text\":").append(jsonString(text))
                        .append("}");
            } catch (Exception e) {
                sb.append("{\"model\":\"").append(modelFile)
                        .append("\",\"error\":").append(jsonString(e.getMessage())).append("}");
            }
        }
        sb.append("]}");
        System.out.println(sb);
    }

    private static String jsonString(String s) {
        if (s == null) { return "null"; }
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r") + "\"";
    }
}

package com.example;

import com.sparrowlogic.whisper4j.WhisperModel;

import java.nio.file.Path;

/**
 * Minimal CLI that transcribes a WAV file and prints each segment as JSONL.
 *
 * Usage: java ... com.example.TranscribeExample model.bin audio.wav
 */
public final class TranscribeExample {

    private TranscribeExample() { }

    public static void main(final String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: TranscribeExample <model-path> <audio.wav>");
            System.exit(1);
        }
        var model = WhisperModel.load(Path.of(args[0]));
        model.transcribe(Path.of(args[1])).forEach(seg -> {
            String escaped = seg.text().replace("\\", "\\\\").replace("\"", "\\\"");
            System.out.printf("{\"id\":%d,\"start\":%.2f,\"end\":%.2f,\"text\":\"%s\"}%n",
                    seg.id(), seg.start(), seg.end(), escaped);
        });
    }
}

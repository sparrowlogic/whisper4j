package com.sparrowlogic.whisper4j.audio;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.logging.Logger;

/**
 * Minimal RIFF/WAV parser using direct ByteBuffer reads.
 * No javax.sound.sampled dependency.
 */
@SuppressWarnings("checkstyle:ExecutableStatementCount")
public final class WavParser {

    private static final Logger LOG = Logger.getLogger(WavParser.class.getName());

    public record WavData(float[] samples, int sampleRate, int channels, int bitsPerSample) {
    }

    private WavParser() {
    }

    /**
     * Parse a WAV file from disk using memory-mapped I/O.
     *
     * @param path path to the WAV file
     * @return parsed audio data with samples, sample rate, and channel info
     * @throws IOException if the file cannot be read
     */
    public static WavData parse(final Path path) throws IOException {
        LOG.info("Parsing WAV file: " + path);
        try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
            var buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            return parse(buf);
        }
    }

    /**
     * Parse a WAV file from a ByteBuffer.
     * Supports PCM (8/16/24/32-bit), IEEE float (32/64-bit), and u-law formats.
     *
     * @param buf little-endian ByteBuffer positioned at the start of the RIFF header
     * @return parsed audio data
     * @throws IllegalArgumentException if the buffer is not a valid WAV file
     */
    @SuppressWarnings("checkstyle:MissingSwitchDefault")
    public static WavData parse(final ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);

        // RIFF header
        int riffTag = buf.getInt();
        if (riffTag != 0x46464952) {
            throw new IllegalArgumentException("not a RIFF file");
        }
        buf.getInt();
        int waveTag = buf.getInt();
        if (waveTag != 0x45564157) {
            throw new IllegalArgumentException("not a WAVE file");
        }

        int audioFormat = 0;
        int channels = 0;
        int sampleRate = 0;
        int bitsPerSample = 0;
        byte[] dataBytes = null;

        // walk chunks
        while (buf.hasRemaining()) {
            int chunkId = buf.getInt();
            int chunkSize = buf.getInt();
            int chunkEnd = buf.position() + chunkSize;

            if (chunkId == 0x20746D66) {
                audioFormat = Short.toUnsignedInt(buf.getShort());
                channels = Short.toUnsignedInt(buf.getShort());
                sampleRate = buf.getInt();
                buf.getInt();
                buf.getShort();
                bitsPerSample = Short.toUnsignedInt(buf.getShort());
            } else if (chunkId == 0x61746164) {
                dataBytes = new byte[chunkSize];
                buf.get(dataBytes);
            }

            buf.position(chunkEnd);
        }

        if (dataBytes == null) {
            throw new IllegalArgumentException("no data chunk");
        }

        float[] samples = switch (audioFormat) {
            case 1 -> decodePcm(dataBytes, bitsPerSample);
            case 3 -> decodeFloat(dataBytes, bitsPerSample);
            case 7 -> decodeUlaw(dataBytes);
            default -> throw new IllegalArgumentException("unsupported format: " + audioFormat);
        };

        LOG.info("WAV parsed: format=%d channels=%d sampleRate=%d bits=%d samples=%d (%.2fs)"
                .formatted(audioFormat, channels, sampleRate, bitsPerSample, samples.length,
                        samples.length / (float) sampleRate / channels));

        return new WavData(samples, sampleRate, channels, bitsPerSample);
    }

    private static float[] decodePcm(final byte[] data, final int bits) {
        return switch (bits) {
            case 8 -> {
                float[] out = new float[data.length];
                for (int i = 0; i < data.length; i++) {
                    out[i] = ((data[i] & 0xFF) - 128) / 128.0f;
                }
                yield out;
            }
            case 16 -> {
                float[] out = new float[data.length / 2];
                for (int i = 0; i < out.length; i++) {
                    short s = (short) ((data[i * 2] & 0xFF) | (data[i * 2 + 1] << 8));
                    out[i] = s / 32768.0f;
                }
                yield out;
            }
            case 24 -> {
                float[] out = new float[data.length / 3];
                for (int i = 0; i < out.length; i++) {
                    int s = (data[i * 3] & 0xFF)
                            | ((data[i * 3 + 1] & 0xFF) << 8) | (data[i * 3 + 2] << 16);
                    out[i] = s / 8388608.0f;
                }
                yield out;
            }
            case 32 -> {
                float[] out = new float[data.length / 4];
                var bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < out.length; i++) {
                    out[i] = bb.getInt() / 2147483648.0f;
                }
                yield out;
            }
            default -> throw new IllegalArgumentException("unsupported PCM bit depth: " + bits);
        };
    }

    private static float[] decodeFloat(final byte[] data, final int bits) {
        var bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        if (bits == 32) {
            float[] out = new float[data.length / 4];
            for (int i = 0; i < out.length; i++) {
                out[i] = bb.getFloat();
            }
            return out;
        } else if (bits == 64) {
            float[] out = new float[data.length / 8];
            for (int i = 0; i < out.length; i++) {
                out[i] = (float) bb.getDouble();
            }
            return out;
        }
        throw new IllegalArgumentException("unsupported float bit depth: " + bits);
    }

    /**
     * u-law decode — ITU-T G.711 table-based, ported from pots-voice UlawCodec.
     */
    private static float[] decodeUlaw(final byte[] data) {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            int ulaw = ~data[i] & 0xFF;
            int sign = (ulaw & 0x80) == 0 ? 1 : -1;
            int exponent = (ulaw & 0x70) >> 4;
            int mantissa = ulaw & 0x0F;
            short sample = (short) (sign * (((2 * mantissa + 33) << (exponent + 2)) - 0x84));
            out[i] = sample / 32768.0f;
        }
        return out;
    }
}

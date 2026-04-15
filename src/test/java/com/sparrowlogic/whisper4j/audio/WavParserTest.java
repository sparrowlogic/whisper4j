package com.sparrowlogic.whisper4j.audio;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

class WavParserTest {

    @Test
    void parse16BitPcmWav() {
        // Build a minimal 16-bit PCM WAV in memory
        short[] samples = {0, 16384, -16384, 32767};
        byte[] wav = buildWav(1, 16000, 16, samples);
        var result = WavParser.parse(ByteBuffer.wrap(wav));

        assertEquals(16000, result.sampleRate());
        assertEquals(1, result.channels());
        assertEquals(samples.length, result.samples().length);
        assertEquals(0, result.samples()[0], 1e-4);
        assertEquals(0.5f, result.samples()[1], 0.01f);
        assertEquals(-0.5f, result.samples()[2], 0.01f);
    }

    @Test
    void parseUlawWav() {
        // format tag 7 = μ-law
        byte[] ulawData = {(byte) 0xFF, (byte) 0x80}; // silence, then a value
        byte[] wav = buildWavRaw(7, 8000, 1, 8, ulawData);
        var result = WavParser.parse(ByteBuffer.wrap(wav));

        assertEquals(8000, result.sampleRate());
        assertEquals(2, result.samples().length);
        assertEquals(0, result.samples()[0], 0.01f); // 0xFF = silence
    }

    @Test
    void parseStereoConvertsCorrectly() {
        short[] samples = {1000, -1000, 2000, -2000};
        byte[] wav = buildWav(2, 44100, 16, samples);
        var result = WavParser.parse(ByteBuffer.wrap(wav));

        assertEquals(2, result.channels());
        assertEquals(4, result.samples().length);
    }

    @Test
    void rejectNonRiff() {
        byte[] bad = "NOT_RIFF_DATA_HERE__".getBytes();
        assertThrows(IllegalArgumentException.class, () -> WavParser.parse(ByteBuffer.wrap(bad)));
    }

    // ---- WAV builder helpers ----

    private static byte[] buildWav(int channels, int sampleRate, int bitsPerSample, short[] samples) {
        byte[] pcm = new byte[samples.length * 2];
        for (int i = 0; i < samples.length; i++) {
            pcm[i * 2] = (byte) (samples[i] & 0xFF);
            pcm[i * 2 + 1] = (byte) ((samples[i] >> 8) & 0xFF);
        }
        return buildWavRaw(1, sampleRate, channels, bitsPerSample, pcm);
    }

    private static byte[] buildWavRaw(int format, int sampleRate, int channels, int bits, byte[] data) {
        int fmtSize = 16;
        int fileSize = 4 + (8 + fmtSize) + (8 + data.length);
        var buf = ByteBuffer.allocate(12 + 8 + fmtSize + 8 + data.length).order(ByteOrder.LITTLE_ENDIAN);
        // RIFF header
        buf.putInt(0x46464952); // "RIFF"
        buf.putInt(fileSize);
        buf.putInt(0x45564157); // "WAVE"
        // fmt chunk
        buf.putInt(0x20746D66); // "fmt "
        buf.putInt(fmtSize);
        buf.putShort((short) format);
        buf.putShort((short) channels);
        buf.putInt(sampleRate);
        buf.putInt(sampleRate * channels * bits / 8); // byte rate
        buf.putShort((short) (channels * bits / 8)); // block align
        buf.putShort((short) bits);
        // data chunk
        buf.putInt(0x61746164); // "data"
        buf.putInt(data.length);
        buf.put(data);
        return buf.array();
    }
}

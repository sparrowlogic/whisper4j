package com.sparrowlogic.whisper4j.tokenizer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WhisperTokenizerTest {

    /** Build a minimal tokenizer.json for testing. */
    private static WhisperTokenizer buildTestTokenizer(boolean multilingual) {
        // Minimal tokenizer JSON with vocab, added_tokens, and merges
        String json = """
                {
                  "model": {
                    "vocab": {
                      "h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6
                    },
                    "merges": ["h e", "l l", "he ll", "hell o"]
                  },
                  "added_tokens": [
                    {"id": 50257, "content": "<|endoftext|>"},
                    {"id": 50258, "content": "<|startoftranscript|>"},
                    {"id": 50259, "content": "<|en|>"},
                    {"id": 50260, "content": "<|fr|>"},
                    {"id": 50261, "content": "<|transcribe|>"},
                    {"id": 50262, "content": "<|translate|>"},
                    {"id": 50263, "content": "<|startoflm|>"},
                    {"id": 50264, "content": "<|startofprev|>"},
                    {"id": 50265, "content": "<|nospeech|>"},
                    {"id": 50266, "content": "<|notimestamps|>"}
                  ]
                }
                """;
        return WhisperTokenizer.fromJson(json, multilingual, "en", "transcribe");
    }

    @Test
    void specialTokenIds() {
        var tok = buildTestTokenizer(true);
        assertEquals(50258, tok.sot());
        assertEquals(50257, tok.eot());
        assertEquals(50266, tok.noTimestamps());
        assertEquals(50265, tok.noSpeech());
        assertEquals(50267, tok.timestampBegin());
    }

    @Test
    void sotSequenceMultilingual() {
        var tok = buildTestTokenizer(true);
        int[] seq = tok.sotSequence();
        assertArrayEquals(new int[]{50258, 50259, 50261}, seq); // SOT, en, transcribe
    }

    @Test
    void sotSequenceEnglishOnly() {
        var tok = buildTestTokenizer(false);
        int[] seq = tok.sotSequence();
        // English-only still includes language + task tokens (whisper.cpp convention)
        assertEquals(50258, seq[0]); // SOT
    }

    @Test
    void decodeSkipsSpecialTokens() {
        var tok = buildTestTokenizer(true);
        // token 4 = "he", token 5 = "ll", token 3 = "o"
        String text = tok.decode(new int[]{4, 5, 3});
        assertEquals("hello", text);
    }

    @Test
    void decodeSkipsEotAndAbove() {
        var tok = buildTestTokenizer(true);
        String text = tok.decode(new int[]{4, 50257, 5}); // eot in middle
        assertEquals("hell", text); // eot is skipped, not a stop signal in decode()
    }

    @Test
    void vocabSizeIncludesAddedTokens() {
        var tok = buildTestTokenizer(true);
        assertTrue(tok.vocabSize() > 10);
    }

    @Test
    void isMultilingual() {
        assertTrue(buildTestTokenizer(true).isMultilingual());
        assertFalse(buildTestTokenizer(false).isMultilingual());
    }
}

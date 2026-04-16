package com.sparrowlogic.whisper4j.tokenizer;

import com.sparrowlogic.whisper4j.annotation.Nullable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Whisper BPE tokenizer. Loads from HuggingFace tokenizer.json format.
 * Handles special tokens (SOT, EOT, timestamps, language, task).
 */
@SuppressWarnings({"checkstyle:ClassFanOutComplexity", "checkstyle:ClassDataAbstractionCoupling"})
public final class WhisperTokenizer {

    private final Map<String, Integer> vocab;

    private final String[] idToToken;

    private final List<String[]> merges;

    private final boolean multilingual;

    private final String languageCode;

    private final String task;

    private final boolean rawVocab; // true if vocab tokens are already UTF-8 (GGML)

    // special token IDs (resolved lazily from vocab)
    private final int sotId;

    private final int eotId;

    private final int transcribeId;

    private final int translateId;

    private final int noTimestampsId;

    private final int noSpeechId;

    @SuppressWarnings("checkstyle:ParameterNumber")
    private WhisperTokenizer(
            final Map<String, Integer> vocab, final String[] idToToken,
            final List<String[]> merges, final boolean multilingual,
            final @Nullable String language, final @Nullable String task,
            final boolean rawVocab
    ) {
        this.vocab = vocab;
        this.idToToken = idToToken;
        this.merges = merges;
        this.multilingual = multilingual;
        this.languageCode = language != null ? language : "en";
        this.task = task != null ? task : "transcribe";
        this.rawVocab = rawVocab;

        this.sotId = this.tokenId("<|startoftranscript|>");
        this.eotId = this.tokenId("<|endoftext|>");
        this.transcribeId = this.tokenId("<|transcribe|>");
        this.translateId = this.tokenId("<|translate|>");
        this.noTimestampsId = this.tokenId("<|notimestamps|>");
        this.noSpeechId = vocab.getOrDefault("<|nospeech|>", vocab.getOrDefault("<|nocaptions|>", -1));
    }

    /**
     * Load a tokenizer from a HuggingFace {@code tokenizer.json} file.
     *
     * @param tokenizerJson path to tokenizer.json
     * @param multilingual  true for multilingual models (vocab &ge; 51865)
     * @param language      ISO 639-1 language code (e.g. "en")
     * @param task          "transcribe" or "translate"
     * @return configured tokenizer
     * @throws IOException if the file cannot be read
     */
    public static WhisperTokenizer load(
            final Path tokenizerJson, final boolean multilingual,
            final String language, final String task
    ) throws IOException {
        String json = Files.readString(tokenizerJson);
        return fromJson(json, multilingual, language, task);
    }

    /**
     * Parse a HuggingFace tokenizer.json. Minimal JSON parsing — no dependency.
     */
    public static WhisperTokenizer fromJson(
            final String json, final boolean multilingual,
            final String language, final String task
    ) {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        List<String[]> merges = new ArrayList<>();

        // extract vocab object
        int vocabStart = json.indexOf("\"vocab\"");
        if (vocabStart >= 0) {
            int braceStart = json.indexOf('{', vocabStart + 7);
            int braceEnd = findMatchingBrace(json, braceStart);
            String vocabJson = json.substring(braceStart + 1, braceEnd);
            parseVocab(vocabJson, vocab);
        }

        // extract added_tokens array for special tokens
        int addedStart = json.indexOf("\"added_tokens\"");
        if (addedStart >= 0) {
            int arrStart = json.indexOf('[', addedStart);
            int arrEnd = findMatchingBracket(json, arrStart);
            String addedJson = json.substring(arrStart + 1, arrEnd);
            parseAddedTokens(addedJson, vocab);
        }

        // extract merges
        int mergesStart = json.indexOf("\"merges\"");
        if (mergesStart >= 0) {
            int arrStart = json.indexOf('[', mergesStart);
            int arrEnd = findMatchingBracket(json, arrStart);
            String mergesJson = json.substring(arrStart + 1, arrEnd);
            parseMerges(mergesJson, merges);
        }

        String[] idToToken = new String[vocab.size()];
        vocab.forEach((token, id) -> {
            if (id >= 0 && id < idToToken.length) {
                idToToken[id] = token;
            }
        });

        return new WhisperTokenizer(vocab, idToToken, merges, multilingual, language, task, false);
    }

    /**
     * Build from GGML vocab where tokens are already raw UTF-8 (not BPE-encoded).
     */
    public static WhisperTokenizer fromRawVocab(
            final String json, final boolean multilingual,
            final String language, final String task
    ) {
        // Parse same as fromJson but mark as raw vocab
        WhisperTokenizer base = fromJson(json, multilingual, language, task);
        return new WhisperTokenizer(
                base.vocab, base.idToToken, base.merges,
                multilingual, language, task, true
        );
    }

    // ---- Public API ----

    /**
     * Start-of-transcript token ID.
     */
    public int sot() {
        return this.sotId;
    }

    /**
     * End-of-text token ID.
     */
    public int eot() {
        return this.eotId;
    }

    public int noTimestamps() {
        return this.noTimestampsId;
    }

    public int noSpeech() {
        return this.noSpeechId;
    }

    public int timestampBegin() {
        return this.noTimestampsId + 1;
    }

    public int[] sotSequence() {
        List<Integer> seq = new ArrayList<>();
        seq.add(this.sotId);
        // Only add language + task tokens for multilingual models (matching Python reference)
        if (this.multilingual) {
            int langId = this.tokenId("<|" + this.languageCode + "|>");
            if (langId >= 0) {
                seq.add(langId);
            }
            seq.add("translate".equals(this.task) ? this.translateId : this.transcribeId);
        }
        return seq.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Encode text to token IDs using byte-level BPE.
     *
     * @param text input text
     * @return array of token IDs
     */
    public int[] encode(final String text) {
        if (text == null || text.isEmpty()) {
            return new int[0];
        }
        // byte-level BPE: convert text to byte tokens, then merge
        List<String> tokens = new ArrayList<>();
        for (byte b : text.getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
            tokens.add(BYTE_ENCODER[b & 0xFF]);
        }
        // apply BPE merges
        this.applyMerges(tokens);
        return tokens.stream().mapToInt(t -> this.vocab.getOrDefault(t, 0)).toArray();
    }

    /**
     * Decode token IDs back to text. Tokens at or above EOT are skipped.
     *
     * @param tokens array of token IDs
     * @return decoded text
     */
    public String decode(final int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int id : tokens) {
            if (id >= this.eotId) {
                continue;
            }
            if (id >= 0 && id < this.idToToken.length && this.idToToken[id] != null) {
                sb.append(this.idToToken[id]);
            }
        }
        return this.rawVocab ? sb.toString() : decodeBpe(sb.toString());
    }

    /**
     * Decode token IDs to text, rendering timestamp tokens as {@code <|0.00|>} markers.
     *
     * @param tokens array of token IDs (may include timestamp tokens)
     * @return decoded text with inline timestamp markers
     */
    public String decodeWithTimestamps(final int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int id : tokens) {
            if (id >= this.timestampBegin()) {
                float ts = (id - this.timestampBegin()) * 0.02f;
                sb.append("<|%.2f|>".formatted(ts));
            } else if (id < this.eotId && id >= 0
                    && id < this.idToToken.length && this.idToToken[id] != null) {
                sb.append(this.idToToken[id]);
            }
        }
        return decodeBpe(sb.toString());
    }

    public Set<Integer> nonSpeechTokens() {
        Set<Integer> result = new TreeSet<>();
        String symbols = "\"#()*+/:;<=>@[\\]^_`{|}~";
        for (char c : symbols.toCharArray()) {
            Integer id = this.vocab.get(String.valueOf(c));
            if (id != null) {
                result.add(id);
            }
            id = this.vocab.get("\u0120" + c);
            if (id != null) {
                result.add(id);
            }
        }
        return result;
    }

    public boolean isMultilingual() {
        return this.multilingual;
    }

    public int vocabSize() {
        return this.vocab.size();
    }

    private int tokenId(final String token) {
        return this.vocab.getOrDefault(token, -1);
    }

    // ---- BPE merge logic ----

    private void applyMerges(final List<String> tokens) {
        for (String[] merge : this.merges) {
            int i = 0;
            while (i < tokens.size() - 1) {
                if (tokens.get(i).equals(merge[0]) && tokens.get(i + 1).equals(merge[1])) {
                    tokens.set(i, merge[0] + merge[1]);
                    tokens.remove(i + 1);
                } else {
                    i++;
                }
            }
        }
    }

    // ---- Byte-level BPE encoding table (GPT-2 style) ----

    private static final String[] BYTE_ENCODER = new String[256];

    private static final Map<Character, Byte> BYTE_DECODER = new HashMap<>();

    static {
        int n = 0;
        char[] mapping = new char[256];
        for (int b = 0; b < 256; b++) {
            if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
                mapping[b] = (char) b;
            } else {
                mapping[b] = (char) (256 + n++);
            }
            BYTE_ENCODER[b] = String.valueOf(mapping[b]);
            BYTE_DECODER.put(mapping[b], (byte) b);
        }
    }

    private static String decodeBpe(final String bpeText) {
        byte[] bytes = new byte[bpeText.length()];
        int len = 0;
        for (int i = 0; i < bpeText.length(); i++) {
            char c = bpeText.charAt(i);
            if (c == '<' && bpeText.startsWith("<|", i)) {
                // pass through timestamp markers
                int end = bpeText.indexOf("|>", i);
                if (end >= 0) {
                    // flush bytes so far
                    String decoded = new String(bytes, 0, len, java.nio.charset.StandardCharsets.UTF_8);
                    len = 0;
                    return decoded + bpeText.substring(i);
                }
            }
            Byte b = BYTE_DECODER.get(c);
            if (b != null) {
                bytes[len++] = b;
            }
        }
        return new String(bytes, 0, len, java.nio.charset.StandardCharsets.UTF_8);
    }

    // ---- Minimal JSON parsing helpers ----

    private static void parseVocab(final String json, final Map<String, Integer> vocab) {
        int pos = 0;
        while (pos < json.length()) {
            int keyStart = json.indexOf('"', pos);
            if (keyStart < 0) {
                break;
            }
            int keyEnd = findStringEnd(json, keyStart);
            String key = unescapeJson(json.substring(keyStart + 1, keyEnd));
            int colon = json.indexOf(':', keyEnd);
            int valStart = colon + 1;
            while (valStart < json.length() && json.charAt(valStart) == ' ') {
                valStart++;
            }
            int valEnd = valStart;
            while (valEnd < json.length()
                    && (Character.isDigit(json.charAt(valEnd)) || json.charAt(valEnd) == '-')) {
                valEnd++;
            }
            vocab.put(key, Integer.parseInt(json.substring(valStart, valEnd).trim()));
            pos = valEnd;
        }
    }

    private static void parseAddedTokens(final String json, final Map<String, Integer> vocab) {
        int pos = 0;
        while (pos < json.length()) {
            int objStart = json.indexOf('{', pos);
            if (objStart < 0) {
                break;
            }
            int objEnd = findMatchingBrace(json, objStart);
            String obj = json.substring(objStart, objEnd + 1);
            String content = extractJsonString(obj, "content");
            Integer id = extractJsonInt(obj, "id");
            if (content != null && id != null) {
                vocab.put(content, id);
            }
            pos = objEnd + 1;
        }
    }

    private static void parseMerges(final String json, final List<String[]> merges) {
        int pos = 0;
        while (pos < json.length()) {
            int strStart = json.indexOf('"', pos);
            if (strStart < 0) {
                break;
            }
            int strEnd = findStringEnd(json, strStart);
            String merge = json.substring(strStart + 1, strEnd);
            String[] parts = merge.split(" ", 2);
            if (parts.length == 2) {
                merges.add(parts);
            }
            pos = strEnd + 1;
        }
    }

    private static int findStringEnd(final String json, final int quoteStart) {
        for (int i = quoteStart + 1; i < json.length(); i++) {
            if (json.charAt(i) == '\\') {
                i++;
                continue;
            }
            if (json.charAt(i) == '"') {
                return i;
            }
        }
        return json.length();
    }

    private static int findMatchingBrace(final String json, final int start) {
        int depth = 0;
        boolean inString = false;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\' && inString) {
                i++;
                continue;
            }
            if (c == '"') {
                inString = !inString;
            }
            if (!inString) {
                if (c == '{') {
                    depth++;
                }
                if (c == '}') {
                    depth--;
                    if (depth == 0) {
                        return i;
                    }
                }
            }
        }
        return json.length() - 1;
    }

    private static int findMatchingBracket(final String json, final int start) {
        int depth = 0;
        boolean inString = false;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\' && inString) {
                i++;
                continue;
            }
            if (c == '"') {
                inString = !inString;
            }
            if (!inString) {
                if (c == '[') {
                    depth++;
                }
                if (c == ']') {
                    depth--;
                    if (depth == 0) {
                        return i;
                    }
                }
            }
        }
        return json.length() - 1;
    }

    private static @Nullable String extractJsonString(final String json, final String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return null;
        }
        int colon = json.indexOf(':', idx + key.length() + 2);
        int strStart = json.indexOf('"', colon + 1);
        if (strStart < 0) {
            return null;
        }
        int strEnd = findStringEnd(json, strStart);
        return unescapeJson(json.substring(strStart + 1, strEnd));
    }

    private static @Nullable Integer extractJsonInt(final String json, final String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return null;
        }
        int colon = json.indexOf(':', idx + key.length() + 2);
        int start = colon + 1;
        while (start < json.length() && json.charAt(start) == ' ') {
            start++;
        }
        int end = start;
        while (end < json.length()
                && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) {
            end++;
        }
        return Integer.parseInt(json.substring(start, end).trim());
    }

    private static String unescapeJson(final String s) {
        return s.replace("\\\"", "\"").replace("\\\\", "\\").replace("\\/", "/")
                .replace("\\n", "\n").replace("\\t", "\t");
    }
}

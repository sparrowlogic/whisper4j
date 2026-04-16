package com.sparrowlogic.whisper4j;

import com.sparrowlogic.whisper4j.audio.FeatureExtractor;
import com.sparrowlogic.whisper4j.audio.Resampler;
import com.sparrowlogic.whisper4j.audio.VoiceActivityDetector;
import java.lang.foreign.Arena;
import com.sparrowlogic.whisper4j.audio.WavParser;
import com.sparrowlogic.whisper4j.model.GgmlLoader;
import com.sparrowlogic.whisper4j.model.ModelDimensions;
import com.sparrowlogic.whisper4j.model.ModelLoader;
import com.sparrowlogic.whisper4j.model.WeightStore;
import com.sparrowlogic.whisper4j.nn.WhisperDecoder;
import com.sparrowlogic.whisper4j.nn.WhisperEncoder;
import com.sparrowlogic.whisper4j.tensor.Tensor;
import com.sparrowlogic.whisper4j.tokenizer.WhisperTokenizer;

import org.jspecify.annotations.Nullable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.Stream;

/**
 * Main entry point for Whisper speech-to-text inference.
 * Pure Java 26 implementation — no native dependencies.
 *
 * Usage:
 *   var model = WhisperModel.load(Path.of("model.pt"));
 *   model.transcribe(Path.of("audio.wav")).forEach(s -> System.out.println(s.text()));
 */
public final class WhisperModel implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(WhisperModel.class.getName());

    private final WhisperEncoder encoder;
    private final WhisperDecoder decoder;
    private final FeatureExtractor featureExtractor;
    private final WhisperTokenizer tokenizer;
    private final ModelDimensions dims;
    private final @Nullable Arena weightArena;
    private final java.util.concurrent.atomic.AtomicInteger activeCount =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private volatile boolean closed;

    private WhisperModel(WhisperEncoder encoder, WhisperDecoder decoder,
                         FeatureExtractor featureExtractor, WhisperTokenizer tokenizer,
                         ModelDimensions dims, @Nullable Arena weightArena) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.featureExtractor = featureExtractor;
        this.tokenizer = tokenizer;
        this.dims = dims;
        this.weightArena = weightArena;
    }

    /**
     * Release off-heap model weight memory.
     * Waits for any in-flight transcriptions to complete before closing.
     * After closing, new transcribe() calls will throw IllegalStateException.
     */
    @Override
    public void close() {
        this.closed = true;
        // Spin-wait for in-flight transcriptions (typically completes in <1s)
        while (this.activeCount.get() > 0) {
            Thread.onSpinWait();
        }
        if (this.weightArena != null) {
            try {
                this.weightArena.close();
            } catch (final IllegalStateException e) {
                LOG.fine("Weight arena already closed: " + e.getMessage());
            }
        }
    }

    private void acquireRef() {
        if (this.closed) {
            throw new IllegalStateException("WhisperModel is closed");
        }
        this.activeCount.incrementAndGet();
        // Double-check after increment to prevent race with close()
        if (this.closed) {
            this.activeCount.decrementAndGet();
            throw new IllegalStateException("WhisperModel is closed");
        }
    }

    private void releaseRef() {
        this.activeCount.decrementAndGet();
    }

    /** Load a Whisper model from any supported format (auto-detected). */
    public static WhisperModel load(Path modelPath) throws IOException {
        return load(modelPath, null, null);
    }

    /**
     * Load a Whisper model with an explicit tokenizer file.
     * @param modelPath  path to .pt, .safetensors, .onnx, GGML .bin, or HuggingFace directory
     * @param tokenizerPath  path to tokenizer.json (or null to auto-discover)
     */
    @SuppressWarnings("checkstyle:CyclomaticComplexity")
    public static WhisperModel load(Path modelPath, @Nullable Path tokenizerPath,
                                     @Nullable TranscriptionOptions opts) throws IOException {
        if (modelPath == null) {
            throw new IllegalArgumentException("modelPath must not be null");
        }
        long t0 = System.currentTimeMillis();
        LOG.info("Loading model from " + modelPath);

        // Resolve HuggingFace directory layout
        Path weightsPath = resolveWeightsPath(modelPath);
        ModelLoader loader = ModelLoader.forPath(weightsPath);
        LOG.info("Detected format: " + loader.getClass().getSimpleName());

        WeightStore weights;
        try {
            weights = loader.load(weightsPath);
        } catch (IOException e) {
            LOG.severe("Failed to load model weights: " + e.getMessage());
            if (loader instanceof GgmlLoader ggml) ggml.close();
            throw e;
        }

        try {
            ModelDimensions dims;
            WhisperTokenizer tokenizer;
            FeatureExtractor fe;

            if (loader instanceof GgmlLoader ggml) {
                dims = ggml.dimensions();
                fe = new FeatureExtractor(dims.nMels(), ggml.melFilters(), ggml.melFilterNFft());
                tokenizer = buildTokenizerFromGgml(ggml.vocab(), dims);
            } else {
                // Normalize weight names from format-specific to canonical
                weights.normalizeNames();
                dims = ModelDimensions.infer(weights);
                fe = new FeatureExtractor(dims.nMels());

                // Auto-discover tokenizer.json alongside model
                Path tokPath = resolveTokenizerPath(tokenizerPath, modelPath);
                if (tokPath != null) {
                    boolean multilingual = dims.nVocab() >= 51865;
                    String lang = opts != null ? opts.language() : "en";
                    String task = opts != null ? opts.task() : "transcribe";
                    tokenizer = WhisperTokenizer.load(tokPath, multilingual, lang, task);
                    LOG.info("Loaded tokenizer from " + tokPath);
                } else {
                    tokenizer = buildDefaultTokenizer(dims);
                    LOG.info("Using default tokenizer (no tokenizer.json found)");
                }
            }

            WhisperEncoder encoder = new WhisperEncoder(weights, dims);
            WhisperDecoder decoder = new WhisperDecoder(weights, dims);
            long elapsed = System.currentTimeMillis() - t0;
            LOG.info("Model loaded in %d ms — %s, %d mels, %d enc layers, %d dec layers, vocab=%d"
                    .formatted(elapsed, dims.nVocab() >= 51865 ? "multilingual" : "english-only",
                        dims.nMels(), dims.nAudioLayer(), dims.nTextLayer(), dims.nVocab()));

            // Transfer weight arena ownership to the model for lifecycle management
            Arena weightArena = (loader instanceof GgmlLoader ggml2) ? ggml2.takeArena() : null;
            return new WhisperModel(encoder, decoder, fe, tokenizer, dims, weightArena);
        } catch (Exception e) {
            LOG.severe("Failed to initialize model: " + e.getMessage());
            if (loader instanceof GgmlLoader ggml) ggml.close();
            throw new IOException("Model initialization failed: " + modelPath, e);
        }
    }

    /**
     * Resolve the actual weights file from a path that may be a HuggingFace directory.
     * HF directories contain: model.safetensors (or pytorch_model.bin), config.json, tokenizer.json
     */
    private static Path resolveWeightsPath(final Path path) {
        if (!Files.isDirectory(path)) {
            return path;
        }
        // HuggingFace directory: prefer safetensors > pytorch > ONNX split > single ONNX
        Path safetensors = path.resolve("model.safetensors");
        if (Files.exists(safetensors)) {
            return safetensors;
        }
        Path pytorch = path.resolve("pytorch_model.bin");
        if (Files.exists(pytorch)) {
            return pytorch;
        }
        // Split ONNX (encoder_model.onnx + decoder_model.onnx)
        if (Files.exists(path.resolve("onnx/encoder_model.onnx"))
                || Files.exists(path.resolve("encoder_model.onnx"))) {
            return path; // OnnxLoader handles directory
        }
        Path onnx = path.resolve("model.onnx");
        if (Files.exists(onnx)) {
            return onnx;
        }
        // CTranslate2 directory (has model.bin + config.json)
        Path ct2 = path.resolve("model.bin");
        if (Files.exists(ct2) && Files.exists(path.resolve("config.json"))) {
            return path;
        }
        Path pt = path.resolve("model.pt");
        if (Files.exists(pt)) {
            return pt;
        }
        throw new IllegalArgumentException("No model file found in directory: " + path);
    }

    /** Auto-discover tokenizer.json alongside the model file or in the same directory. */
    private static @Nullable Path resolveTokenizerPath(final @Nullable Path explicit,
                                                        final Path modelPath) {
        if (explicit != null && Files.exists(explicit)) {
            return explicit;
        }
        Path dir = Files.isDirectory(modelPath) ? modelPath : modelPath.getParent();
        if (dir == null) {
            return null;
        }
        Path candidate = dir.resolve("tokenizer.json");
        return Files.exists(candidate) ? candidate : null;
    }

    // ---- Transcription API ----

    /** Transcribe a WAV file. Returns a lazy stream of segments. */
    public Stream<Segment> transcribe(Path audioPath) throws IOException {
        WavParser.WavData wav = WavParser.parse(audioPath);
        float[] mono16k = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        return transcribe(mono16k);
    }

    /** Transcribe a WAV file with options. */
    public Stream<Segment> transcribe(Path audioPath, TranscriptionOptions opts) throws IOException {
        WavParser.WavData wav = WavParser.parse(audioPath);
        float[] mono16k = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());
        return transcribe(mono16k, opts);
    }

    /** Transcribe raw 16kHz mono float PCM. Returns a lazy stream of segments. */
    public Stream<Segment> transcribe(float[] audio) {
        return transcribe(audio, new TranscriptionOptions());
    }

    /** Transcribe with options. Uses seek-based chunking with timestamp tokens when enabled. */
    @SuppressWarnings({"checkstyle:ExecutableStatementCount", "checkstyle:CyclomaticComplexity",
        "checkstyle:NPathComplexity", "checkstyle:MethodLength"})
    public Stream<Segment> transcribe(float[] audio, TranscriptionOptions opts) {
        return transcribe(audio, opts, null);
    }

    /**
     * Transcribe with cancellation support.
     * @param handle if non-null, checked between chunks — cancel() stops early with partial results
     */
    @SuppressWarnings({"checkstyle:ExecutableStatementCount", "checkstyle:CyclomaticComplexity",
        "checkstyle:NPathComplexity", "checkstyle:MethodLength"})
    public Stream<Segment> transcribe(float[] audio, TranscriptionOptions opts,
                                      @Nullable TranscriptionHandle handle) {
        if (audio == null || audio.length == 0) {
            return Stream.empty();
        }
        if (opts == null) {
            opts = new TranscriptionOptions();
        }
        this.acquireRef();
        try {
            float duration = audio.length / 16000.0f;
            LOG.info("Transcribing %.1f seconds of audio (language=%s, task=%s)".formatted(
                    duration, opts.language(), opts.task()));

            List<Segment> result;
            if (opts.vadFilter()) {
                result = this.transcribeWithVad(audio, opts, duration, handle).toList();
            } else {
                result = this.transcribeChunked(audio, opts, duration, handle).toList();
            }
            return result.stream();
        } finally {
            this.releaseRef();
        }
    }

    private Stream<Segment> transcribeWithVad(final float[] audio, final TranscriptionOptions opts,
                                              final float duration,
                                              final @Nullable TranscriptionHandle handle) {
        var vad = new VoiceActivityDetector();
        var speechSegments = vad.detect(audio);
        LOG.info("VAD: %d speech segments detected".formatted(speechSegments.size()));

        List<Segment> segments = new ArrayList<>();
        int segId = 0;
        int[] suppressTokens = this.buildSuppressTokens();

        for (var speech : speechSegments) {
            if (handle != null && handle.isCancelled()) { break; }
            int len = speech.endSample() - speech.startSample();
            // Pad to 30s for mel extraction
            float[] chunk = new float[featureExtractor.nSamples()];
            System.arraycopy(audio, speech.startSample(), chunk, 0, Math.min(len, chunk.length));

            Tensor features = featureExtractor.extract(chunk);
            features = featureExtractor.padOrTrim(features);
            features = features.reshape(1, dims.nMels(), featureExtractor.maxFrames());

            Tensor encoderOutput = encoder.forward(features);

            int[] prompt = this.buildPrompt(opts);
            int maxTokens = dims.nTextCtx() / 2;
            int contentFrames = (int) (len / 160.0f);
            int fullFrames = featureExtractor.maxFrames();
            if (contentFrames < fullFrames) {
                maxTokens = Math.max(8, maxTokens * contentFrames / fullFrames);
            }
            WhisperDecoder.DecodeResult result = decoder.decodeWithFallback(
                    encoderOutput, prompt, tokenizer.eot(), maxTokens,
                    suppressTokens, tokenizer.noSpeech(), -1, opts.beamSize());

            String text = tokenizer.decode(result.tokens()).trim();
            if (!text.isEmpty()) {
                segments.add(new Segment(++segId, speech.startSeconds(16000),
                        Math.min(speech.endSeconds(16000), duration), text, result.tokens()));
            }
        }
        return segments.stream();
    }

    @SuppressWarnings({"checkstyle:ExecutableStatementCount", "checkstyle:CyclomaticComplexity",
        "checkstyle:NPathComplexity", "checkstyle:MethodLength"})
    private Stream<Segment> transcribeChunked(final float[] audio, final TranscriptionOptions opts,
                                              final float duration,
                                              final @Nullable TranscriptionHandle handle) {
        int nFrames = featureExtractor.maxFrames(); // 3000
        int inputStride = nFrames / dims.nAudioCtx(); // 2
        float timePrecision = inputStride * 160.0f / 16000.0f; // 0.02s
        int chunkSamples = featureExtractor.nSamples();
        int totalFrames = (int) (audio.length / 160.0f);
        int[] suppressTokens = this.buildSuppressTokens();
        int tsBegin = opts.withTimestamps() ? tokenizer.timestampBegin() : -1;

        List<Segment> segments = new ArrayList<>();
        int segId = 0;
        int seekFrame = 0;
        int[] previousTokens = new int[0];

        while (seekFrame < totalFrames) {
            if (handle != null && handle.isCancelled()) { break; }
            float timeOffset = seekFrame * 160.0f / 16000.0f;
            int segmentSize = Math.min(nFrames, totalFrames - seekFrame);

            // Extract mel for this seek position
            int audioStart = seekFrame * 160;
            int audioEnd = Math.min(audioStart + chunkSamples, audio.length);
            float[] chunk = new float[chunkSamples];
            System.arraycopy(audio, audioStart, chunk, 0, Math.min(audioEnd - audioStart, chunkSamples));

            Tensor features = featureExtractor.extract(chunk);
            features = featureExtractor.padOrTrim(features);
            features = features.reshape(1, dims.nMels(), nFrames);

            long encStart = System.currentTimeMillis();
            Tensor encoderOutput = encoder.forward(features);
            long encMs = System.currentTimeMillis() - encStart;

            // Build prompt with optional previous context
            int[] prompt = this.buildPromptWithContext(opts, previousTokens);

            int maxTokens = dims.nTextCtx() / 2;
            if (segmentSize < nFrames) {
                maxTokens = Math.max(8, maxTokens * segmentSize / nFrames);
            }
            long decStart = System.currentTimeMillis();
            WhisperDecoder.DecodeResult result = decoder.decodeWithFallback(
                    encoderOutput, prompt, tokenizer.eot(), maxTokens,
                    suppressTokens, tokenizer.noSpeech(), tsBegin, opts.beamSize());
            long decMs = System.currentTimeMillis() - decStart;
            LOG.info("Chunk at %.1fs: encoder=%dms decoder=%dms".formatted(timeOffset, encMs, decMs));

            // If conditionOnPrevious caused hallucination, retry without context
            if (opts.conditionOnPrevious() && previousTokens.length > 0
                    && result.avgLogprob() < -1.0f) {
                prompt = this.buildPrompt(opts);
                result = decoder.decodeWithFallback(
                        encoderOutput, prompt, tokenizer.eot(), maxTokens,
                        suppressTokens, tokenizer.noSpeech(), tsBegin, opts.beamSize());
                previousTokens = new int[0]; // reset context
            }

            int[] tokens = result.tokens();

            // No-speech check
            if (result.noSpeechProb() > 0.6f && result.avgLogprob() < -1.0f) {
                seekFrame += segmentSize;
                continue;
            }

            // Parse timestamp tokens to create sub-segments
            if (opts.withTimestamps() && tsBegin > 0) {
                segId = this.parseTimestampSegments(tokens, tsBegin, timeOffset, timePrecision,
                        result, segments, segId);
                // Advance seek based on last timestamp
                int lastTsPos = this.findLastTimestampPos(tokens, tsBegin);
                if (lastTsPos > 0) {
                    int advance = lastTsPos * inputStride;
                    seekFrame += Math.max(advance, inputStride);
                } else {
                    seekFrame += segmentSize;
                }
            } else {
                // No timestamps — one segment per chunk
                String text = tokenizer.decode(tokens).trim();
                if (!text.isEmpty()) {
                    float segEnd = Math.min(timeOffset + segmentSize * 160.0f / 16000.0f, duration);
                    segments.add(new Segment(++segId, timeOffset, segEnd, text, tokens));
                }
                seekFrame += segmentSize;
            }

            // Condition on previous text
            if (opts.conditionOnPrevious()) {
                previousTokens = tokens;
            }
        }

        return segments.stream();
    }

    /** Parse timestamp tokens into sub-segments within a 30s chunk. */
    private int parseTimestampSegments(final int[] tokens, final int tsBegin,
                                       final float timeOffset, final float timePrecision,
                                       final WhisperDecoder.DecodeResult result,
                                       final List<Segment> segments, int segId) {
        // Find consecutive timestamp pairs
        List<Integer> tsPositions = new ArrayList<>();
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] >= tsBegin) {
                tsPositions.add(i);
            }
        }

        if (tsPositions.size() < 2) {
            // No timestamp pairs — treat as single segment
            int[] textTokens = filterTextTokens(tokens, tsBegin);
            String text = tokenizer.decode(textTokens).trim();
            if (!text.isEmpty()) {
                float start = timeOffset;
                float end = tsPositions.isEmpty() ? timeOffset + 30.0f
                        : timeOffset + (tokens[tsPositions.getFirst()] - tsBegin) * timePrecision;
                segments.add(new Segment(++segId, start, end, text, textTokens));
            }
            return segId;
        }

        // Split at consecutive timestamp pairs (matching Python whisper transcribe.py)
        int lastSlice = 0;
        for (int i = 0; i < tsPositions.size() - 1; i++) {
            if (tsPositions.get(i + 1) == tsPositions.get(i) + 1) {
                // Consecutive timestamps found — slice up to (not including) the second timestamp
                int sliceEnd = tsPositions.get(i + 1);
                int[] slice = new int[sliceEnd - lastSlice];
                System.arraycopy(tokens, lastSlice, slice, 0, slice.length);

                // Find first and last timestamp tokens in the slice for start/end times
                float start = timeOffset;
                float end = timeOffset + 30.0f;
                for (int j = 0; j < slice.length; j++) {
                    if (slice[j] >= tsBegin) {
                        start = timeOffset + (slice[j] - tsBegin) * timePrecision;
                        break;
                    }
                }
                for (int j = slice.length - 1; j >= 0; j--) {
                    if (slice[j] >= tsBegin) {
                        end = timeOffset + (slice[j] - tsBegin) * timePrecision;
                        break;
                    }
                }

                int[] textTokens = filterTextTokens(slice, tsBegin);
                String text = tokenizer.decode(textTokens).trim();
                if (!text.isEmpty()) {
                    segments.add(new Segment(++segId, start, end, text, textTokens));
                }
                lastSlice = sliceEnd;
            }
        }
        return segId;
    }

    private int findLastTimestampPos(final int[] tokens, final int tsBegin) {
        for (int i = tokens.length - 1; i >= 0; i--) {
            if (tokens[i] >= tsBegin) {
                return tokens[i] - tsBegin;
            }
        }
        return -1;
    }

    private static int[] filterTextTokens(final int[] tokens, final int tsBegin) {
        return java.util.Arrays.stream(tokens).filter(t -> t < tsBegin).toArray();
    }

    private int[] buildPromptWithContext(final TranscriptionOptions opts, final int[] previousTokens) {
        int[] sot = this.buildPrompt(opts);
        if (!opts.conditionOnPrevious() || previousTokens.length == 0) {
            return sot;
        }
        // Prepend previous tokens as context, capped so total prompt fits in nTextCtx/2
        int maxPrompt = dims.nTextCtx() / 2 - 1;
        int maxPrev = maxPrompt - sot.length - 1; // -1 for sotPrev token
        if (maxPrev <= 0) {
            return sot;
        }
        int prevLen = Math.min(previousTokens.length, maxPrev);
        int[] prevSlice = new int[prevLen];
        System.arraycopy(previousTokens, previousTokens.length - prevLen, prevSlice, 0, prevLen);

        int numLangs = dims.nVocab() - 51765 - (isMultilingual() ? 1 : 0);
        int dt = numLangs - 98;
        int sotPrev = 50360 + dt;

        int[] result = new int[1 + prevLen + sot.length];
        result[0] = sotPrev;
        System.arraycopy(prevSlice, 0, result, 1, prevLen);
        System.arraycopy(sot, 0, result, 1 + prevLen, sot.length);
        return result;
    }

    /** Detect the spoken language from the first 30s of audio. */
    public String detectLanguage(float[] audio) {
        if (audio == null || audio.length == 0) { return "en"; }
        if (!this.isMultilingual()) { return "en"; }
        this.acquireRef();
        try {
            float[] chunk = new float[featureExtractor.nSamples()];
        System.arraycopy(audio, 0, chunk, 0, Math.min(audio.length, chunk.length));
        Tensor features = featureExtractor.extract(chunk);
        features = featureExtractor.padOrTrim(features);
        features = features.reshape(1, dims.nMels(), featureExtractor.maxFrames());
        Tensor encoderOutput = encoder.forward(features);

        // Run decoder with just SOT token, get logits at SOT position
        int[] sotOnly = {tokenizer.sot()};
        Tensor logits = decoder.forward(sotOnly, encoderOutput, null);
        float[] sotLogits = new float[dims.nVocab()];
        System.arraycopy(logits.data(), 0, sotLogits, 0, dims.nVocab());

        // Find best language token (50258..50258+numLangs)
        int langBase = 50258;
        int numLangs = dims.nVocab() - 51765 - (isMultilingual() ? 1 : 0);
        int bestLang = 0;
        float bestScore = sotLogits[langBase];
        for (int i = 1; i < numLangs; i++) {
            if (sotLogits[langBase + i] > bestScore) {
                bestScore = sotLogits[langBase + i];
                bestLang = i;
            }
        }
        String[] langs = {"en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv","it","id","hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn","bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc","ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn","mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su","yue"};
        String detected = bestLang < langs.length ? langs[bestLang] : "en";
        LOG.info("Detected language: " + detected);
        return detected;
        } finally {
            this.releaseRef();
        }
    }

    /** Build suppress token list matching whisper reference. */
    private int[] buildPrompt(final TranscriptionOptions opts) {
        int[] prompt = tokenizer.sotSequence();
        if (!opts.withTimestamps()) {
            int[] withNoTs = new int[prompt.length + 1];
            System.arraycopy(prompt, 0, withNoTs, 0, prompt.length);
            withNoTs[prompt.length] = tokenizer.noTimestamps();
            prompt = withNoTs;
        }
        return prompt;
    }

    /** Build suppress token list matching whisper reference (decoding.py _get_suppress_tokens). */
    private int[] buildSuppressTokens() {
        List<Integer> suppress = new ArrayList<>();
        // Non-speech tokens: symbols, annotations, music notes (matching whisper reference)
        // Generated from tokenizer.non_speech_tokens using GPT-2 BPE encoding
        int[] nonSpeech = {1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 63,
                90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279,
                1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211,
                4600, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907,
                13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724,
                22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282,
                49146};
        for (int t : nonSpeech) { suppress.add(t); }
        // Suppress special tokens: SOT, translate, transcribe, SOT_prev, SOT_lm, no_speech
        suppress.add(tokenizer.sot());
        int numLangs = dims.nVocab() - 51765 - (isMultilingual() ? 1 : 0);
        int dt = numLangs - 98;
        suppress.add(50357 + dt); // translate
        suppress.add(50358 + dt); // transcribe
        suppress.add(50359 + dt); // startoflm
        suppress.add(50360 + dt); // startofprev
        if (tokenizer.noSpeech() >= 0) { suppress.add(tokenizer.noSpeech()); }
        return suppress.stream().mapToInt(Integer::intValue).toArray();
    }

    public ModelDimensions dimensions() { return dims; }
    public boolean isMultilingual() { return dims.nVocab() >= 51865; }

    // ---- Records ----

    public record Segment(int id, float start, float end, String text, int[] tokens) { }

    /** Word-level timestamp, estimated from token positions within a segment. */
    public record Word(float start, float end, String word) { }

    /** Handle for monitoring or cancelling an in-flight transcription. */
    public static final class TranscriptionHandle {
        private final java.util.concurrent.atomic.AtomicBoolean cancelled =
                new java.util.concurrent.atomic.AtomicBoolean(false);

        /** Cancel the transcription. Returns partial results collected so far. */
        public void cancel() {
            this.cancelled.set(true);
        }

        /** Check if cancellation was requested. */
        public boolean isCancelled() {
            return this.cancelled.get();
        }
    }

    /** Extract word-level timestamps from a segment by distributing time proportionally. */
    public static List<Word> wordTimestamps(final Segment segment) {
        String text = segment.text().trim();
        if (text.isEmpty()) {
            return List.of();
        }
        // Split on word boundaries (spaces, or before uppercase after lowercase)
        String[] rawWords = text.split("(?<=\\S)(?=\\s)|(?<=\\s)(?=\\S)");
        List<String> words = new ArrayList<>();
        for (String w : rawWords) {
            if (!w.isBlank()) {
                words.add(w);
            }
        }
        if (words.isEmpty()) {
            return List.of();
        }

        float totalDuration = segment.end() - segment.start();
        int totalChars = words.stream().mapToInt(String::length).sum();
        float timePerChar = totalDuration / Math.max(1, totalChars);
        List<Word> result = new ArrayList<>();
        float pos = segment.start();
        for (String w : words) {
            float wordDuration = w.length() * timePerChar;
            result.add(new Word(pos, pos + wordDuration, w));
            pos += wordDuration;
        }

        mergePunctuations(result);
        return result;
    }

    /**
     * Merge punctuation with adjacent words, matching whisper reference timing.py.
     * Prepended punctuation (quotes, brackets) merges with following word.
     * Appended punctuation (periods, commas) merges with preceding word.
     */
    private static void mergePunctuations(final List<Word> words) {
        String prepended = "\"'\u201c\u00bf([{-";
        String appended = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}、";

        // Merge prepended punctuation with following word
        for (int i = words.size() - 2; i >= 0; i--) {
            Word prev = words.get(i);
            if (prepended.indexOf(prev.word().trim().isEmpty() ? ' ' : prev.word().trim().charAt(0)) >= 0
                    && prev.word().trim().length() == 1) {
                Word next = words.get(i + 1);
                words.set(i + 1, new Word(prev.start(), next.end(), prev.word() + next.word()));
                words.set(i, new Word(prev.start(), prev.start(), ""));
            }
        }

        // Merge appended punctuation with preceding word
        for (int j = 1; j < words.size(); j++) {
            Word curr = words.get(j);
            if (curr.word().length() == 1 && appended.indexOf(curr.word().charAt(0)) >= 0) {
                Word prev = words.get(j - 1);
                words.set(j - 1, new Word(prev.start(), curr.end(), prev.word() + curr.word()));
                words.set(j, new Word(curr.end(), curr.end(), ""));
            }
        }

        // Remove empty entries
        words.removeIf(w -> w.word().isEmpty());
    }

    public record TranscriptionOptions(
            String language,
            String task,
            int beamSize,
            boolean withTimestamps,
            boolean vadFilter,
            boolean conditionOnPrevious
    ) {
        public TranscriptionOptions() {
            this("en", "transcribe", 5, true, false, false);
        }
    }

    // ---- Internal ----

    private static WhisperTokenizer buildTokenizerFromGgml(Map<Integer, String> vocab, ModelDimensions dims) {
        boolean multilingual = dims.nVocab() >= 51865;
        int numLangs = dims.nVocab() - 51765 - (multilingual ? 1 : 0);
        int dt = numLangs - 98;
        // Inject special tokens at correct positions (whisper.cpp convention)
        vocab.put(50256, "<|endoftext|>");
        vocab.put(50257, "<|startoftranscript|>");
        String[] langs = {"en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv","it","id","hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn","bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc","ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn","mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su","yue"};
        for (int i = 0; i < Math.min(langs.length, numLangs); i++) {
            vocab.put(50258 + i, "<|" + langs[i] + "|>");
        }
        vocab.put(50357 + dt, "<|translate|>");
        vocab.put(50358 + dt, "<|transcribe|>");
        vocab.put(50359 + dt, "<|startoflm|>");
        vocab.put(50360 + dt, "<|startofprev|>");
        vocab.put(50361 + dt, "<|nospeech|>");
        vocab.put(50362 + dt, "<|notimestamps|>");
        return buildTokenizerFromVocab(vocab, dims);
    }

    private static WhisperTokenizer buildTokenizerFromVocab(Map<Integer, String> vocab, ModelDimensions dims) {
        boolean multilingual = dims.nVocab() >= 51865;
        // Build JSON from the GGML vocab — it already has all tokens including specials
        StringBuilder json = new StringBuilder("{\"model\":{\"vocab\":{");
        boolean first = true;
        for (var entry : vocab.entrySet()) {
            if (!first) json.append(',');
            first = false;
            String escaped = entry.getValue().replace("\\", "\\\\").replace("\"", "\\\"");
            json.append('"').append(escaped).append("\":").append(entry.getKey());
        }
        json.append("},\"merges\":[]},\"added_tokens\":[");
        // add special tokens that may already be in vocab but need to be in added_tokens for the tokenizer
        first = true;
        for (var entry : vocab.entrySet()) {
            String token = entry.getValue();
            if (token.startsWith("<|") && token.endsWith("|>")) {
                if (!first) json.append(',');
                first = false;
                json.append("{\"id\":").append(entry.getKey())
                        .append(",\"content\":\"").append(token.replace("\"", "\\\"")).append("\"}");
            }
        }
        json.append("]}");
        return WhisperTokenizer.fromRawVocab(json.toString(), multilingual, "en", "transcribe");
    }

    private static WhisperTokenizer buildDefaultTokenizer(ModelDimensions dims) {
        boolean multilingual = dims.nVocab() >= 51865;
        // whisper.cpp: base IDs assume 98 languages, then adjusts by dt = numLangs - 98
        int numLangs = dims.nVocab() - 51765 - (multilingual ? 1 : 0);
        int dt = numLangs - 98;
        int eot = 50256;
        int sot = 50257;
        int translate  = 50357 + dt;
        int transcribe = 50358 + dt;
        int solm       = 50359 + dt;
        int prev       = 50360 + dt;
        int nospeech   = 50361 + dt;
        int notimestamps = 50362 + dt;

        String[] langs = {"en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv","it","id","hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn","bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc","ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn","mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su","yue"};
        StringBuilder json = new StringBuilder("{\"model\":{\"vocab\":{},\"merges\":[]},\"added_tokens\":[");
        json.append("{\"id\":").append(eot).append(",\"content\":\"<|endoftext|>\"},");
        json.append("{\"id\":").append(sot).append(",\"content\":\"<|startoftranscript|>\"},");
        for (int i = 0; i < Math.min(langs.length, numLangs); i++) {
            json.append("{\"id\":").append(50258 + i).append(",\"content\":\"<|").append(langs[i]).append("|>\"},");
        }
        json.append("{\"id\":").append(translate).append(",\"content\":\"<|translate|>\"},");
        json.append("{\"id\":").append(transcribe).append(",\"content\":\"<|transcribe|>\"},");
        json.append("{\"id\":").append(solm).append(",\"content\":\"<|startoflm|>\"},");
        json.append("{\"id\":").append(prev).append(",\"content\":\"<|startofprev|>\"},");
        json.append("{\"id\":").append(nospeech).append(",\"content\":\"<|nospeech|>\"},");
        json.append("{\"id\":").append(notimestamps).append(",\"content\":\"<|notimestamps|>\"}");
        json.append("]}");
        return WhisperTokenizer.fromJson(json.toString(), multilingual, "en", "transcribe");
    }
}

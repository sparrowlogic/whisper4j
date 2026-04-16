# whisper4j

Pure Java implementation of [OpenAI Whisper](https://github.com/openai/whisper) speech-to-text.
Zero native dependencies — runs on any JVM 25 or higher.
Automatically uses hardware acceleration when available (tested on Apple Silicon M2 Max).

## Project motivations, goals, and expectations.

**Motivations:**
- To bring more AI/ML capabilities natively into the Java ecosystem.
- To understand the Vector API in Java 26+ (Preview).
- To understand how the python implementation performs in comparison to Java.

**Goals:**
- Get to parity or better performance in comparison to the python implementation.

**Expectations:**
- It's not currently equivalent in performance to the original python implementation that relies on optimized
  dependencies.
- I'm not sure if this can achieve equivalent or better performance than the reference implementation.
- If you identify optimization opportunities, please submit a PR and help improve this project!

## Quick Start

### Maven Dependency

```xml
<dependency>
    <groupId>com.sparrowlogic</groupId>
    <artifactId>whisper4j</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
```

SNAPSHOT versions require the Central snapshots repository:

```xml
<repositories>
    <repository>
        <id>central-snapshots</id>
        <url>https://central.sonatype.com/repository/maven-snapshots/</url>
        <snapshots><enabled>true</enabled></snapshots>
    </repository>
</repositories>
```

### Usage

1. Download a model from [HuggingFace](https://huggingface.co/models?search=whisper)
   or [OpenAI](https://openai.com/blog/whisper) into the `./models/` directory for local development.
2. Be the best Java dev you can be.

```java
var model = WhisperModel.load(Path.of("ggml-base.en.bin"));

// Transcribe a file
model.transcribe(Path.of("audio.wav")).forEach(seg ->
        System.out.printf("[%.1f - %.1f] %s%n", seg.start(), seg.end(), seg.text()));

// Transcribe raw PCM
model.transcribe(float16kHzMono).forEach(seg -> System.out.println(seg.text()));

// Detect language
String lang = model.detectLanguage(audio); // "en", "zh", "de", etc.
```

```bash
# Java 25+ (base — no extra flags needed)
java --enable-native-access=ALL-UNNAMED -jar app.jar

# Java 26+ (optional — enables Vector API SIMD acceleration)
java --enable-preview --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED -jar app.jar
```

### Factory / Dependency Injection

Use `WhisperModelFactory` for Spring Boot, Guice, or any DI framework:

```java
// Spring Boot @Configuration
@Bean
public WhisperModel whisperModel(@Value("${whisper.model-path}") String path) {
    return new WhisperModelFactory(Path.of(path))
            .language("en")
            .beamSize(5)
            .withTimestamps(true)
            .create();  // thread-safe — inject as singleton
}
```

```java
// Plain Java
var model = new WhisperModelFactory(Path.of("ggml-base.en.bin"))
        .language("en")
        .beamSize(1)
        .create();
```


## API Comparison: whisper4j vs OpenAI Whisper (Python)

### Loading a Model

| Python                                            | Java                                                                 |
|---------------------------------------------------|----------------------------------------------------------------------|
| `model = whisper.load_model("base.en")`           | `var model = WhisperModel.load(Path.of("ggml-base.en.bin"));`        |
| `model = whisper.load_model("large-v3-turbo")`    | `var model = WhisperModel.load(Path.of("ggml-large-v3-turbo.bin"));` |
| `model = whisper.load_model("/path/to/model.pt")` | `var model = WhisperModel.load(Path.of("/path/to/model.bin"));`      |

### Transcription

**Python:**

```python
result = whisper.transcribe(
    model, "audio.wav",
    language="en",
    task="transcribe",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=True,
    word_timestamps=False,
    initial_prompt=None,
)
for segment in result["segments"]:
    print(f"[{segment['start']:.1f} - {segment['end']:.1f}] {segment['text']}")
```

**Java:**

```java
var opts = new TranscriptionOptions(
        "en",           // language
        "transcribe",   // task
        1,              // beamSize
        false,          // withTimestamps
        false,          // vadFilter
        false           // conditionOnPrevious
);
model.

transcribe(Path.of("audio.wav"),opts).

forEach(seg ->
        System.out.

printf("[%.1f - %.1f] %s%n",seg.start(),seg.

end(),seg.

text()));
```

**Default (no options):**

```java
// Uses: language=en, task=transcribe, beam_size=5, timestamps=true,
// temperature fallback, compression_ratio_threshold=2.4,
// logprob_threshold=-1.0, no_speech_threshold=0.6
model.transcribe(Path.of("audio.wav")).

forEach(seg ->
        System.out.

println(seg.text()));
```

### TranscriptionOptions

| Python Parameter              | Java Parameter        | Default        | Notes                              |
|-------------------------------|-----------------------|----------------|------------------------------------|
| `language`                    | `language`            | `"en"`         | ISO 639-1 code                     |
| `task`                        | `task`                | `"transcribe"` | `"transcribe"` or `"translate"`    |
| `beam_size`                   | `beamSize`            | `5`            | 1 = greedy, 5 = beam search        |
| `word_timestamps`             | `withTimestamps`      | `true`         | Timestamp token generation         |
| —                             | `vadFilter`           | `false`        | Energy-based VAD (whisper4j extra) |
| `condition_on_previous_text`  | `conditionOnPrevious` | `false`        | Cross-chunk context                |
| `temperature`                 | —                     | `(0.0..1.0)`   | Always uses fallback sequence      |
| `compression_ratio_threshold` | —                     | `2.4`          | Built-in, not configurable         |
| `logprob_threshold`           | —                     | `-1.0`         | Built-in, not configurable         |
| `no_speech_threshold`         | —                     | `0.6`          | Built-in, not configurable         |
| `initial_prompt`              | —                     | —              | Not yet implemented                |

### Return Types

**Python:**

```python
result = {
    "text": "full transcription...",
    "segments": [
        {"id": 0, "start": 0.0, "end": 5.0, "text": "...", "tokens": [...],
         "temperature": 0.0, "avg_logprob": -0.3, "compression_ratio": 1.2,
         "no_speech_prob": 0.01}
    ],
    "language": "en"
}
```

**Java:**

```java
Stream<Segment> segments = model.transcribe(audio);
// Segment(int id, float start, float end, String text, int[] tokens)

List<Word> words = WhisperModel.wordTimestamps(segment);
// Word(float start, float end, String word)
```

### Language Detection

| Python                                     | Java                                                                                        |
|--------------------------------------------|---------------------------------------------------------------------------------------------|
| `audio = whisper.load_audio("audio.wav")`  | `var wav = WavParser.parse(Path.of("audio.wav"));`                                          |
| `mel = whisper.log_mel_spectrogram(audio)` | `float[] mono = Resampler.toWhisperInput(wav.samples(), wav.sampleRate(), wav.channels());` |
| `_, probs = model.detect_language(mel)`    | `String lang = model.detectLanguage(mono);`                                                 |

### Word Timestamps

| Python                                                            | Java                                                |
|-------------------------------------------------------------------|-----------------------------------------------------|
| `result = whisper.transcribe(model, audio, word_timestamps=True)` | `var words = WhisperModel.wordTimestamps(segment);` |
| `word["word"], word["start"], word["end"]`                        | `word.word(), word.start(), word.end()`             |

## Feature Parity Matrix

| Feature                   | Python whisper     | whisper4j | Status                           |
|---------------------------|--------------------|-----------|----------------------------------|
| Greedy decoding           | ✅                  | ✅         | Parity                           |
| Temperature fallback      | ✅                  | ✅         | Parity                           |
| Compression ratio check   | ✅                  | ✅         | Parity                           |
| No-speech detection       | ✅                  | ✅         | Parity                           |
| Token suppression         | ✅                  | ✅         | Parity                           |
| Blank suppression         | ✅                  | ✅         | Parity                           |
| Timestamp tokens          | ✅                  | ✅         | Parity                           |
| Seek-based chunking       | ✅                  | ✅         | Parity                           |
| Language detection        | ✅                  | ✅         | Parity                           |
| Punctuation merging       | ✅                  | ✅         | Parity                           |
| DTW word alignment        | ✅                  | ✅         | Implemented                      |
| N-gram repetition penalty | ❌                  | ✅         | whisper4j extra                  |
| VAD filtering             | ❌ (faster-whisper) | ✅         | whisper4j extra                  |
| Beam search               | ✅                  | ✅         | Parity                           |
| initial_prompt            | ✅                  | ❌         | Not yet                          |
| condition_on_previous     | ✅                  | 🟡        | Implemented, disabled by default |
| Text normalizers          | ✅                  | ❌         | Not needed for transcription     |
| CUDA/GPU                  | ✅                  | ❌         | JVM — uses BLAS instead          |

## Supported Model Formats

| Format                       | Extension                   | Status         | Notes                                                                                                                                                                                            |
|------------------------------|-----------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GGML (whisper.cpp)           | `.bin`                      | ✅ Full         | Primary format, mmap loading, zero-copy F32                                                                                                                                                      |
| SafeTensors (HuggingFace)    | `.safetensors`              | ✅ Full         | File or HF directory, auto tokenizer discovery                                                                                                                                                   |
| PyTorch                      | `.pt` / `pytorch_model.bin` | ✅ Full         | ZIP + pickle interpreter, HF format supported                                                                                                                                                    |
| ONNX                         | `.onnx`                     | 🟡 Partial     | Named initializers work; HF community models use opaque weight names requiring graph tracing (not yet supported)                                                                                 |
| CTranslate2 (faster-whisper) | `model.bin`                 | 🟡 Loader only | Binary parser implemented but weight name mapping from CT2 naming to Whisper canonical names not yet validated. Models available from [Systran/faster-whisper-*](https://huggingface.co/Systran) |

HuggingFace directory layout is auto-detected — pass a directory containing `model.safetensors`, `pytorch_model.bin`, or
`encoder_model.onnx` and whisper4j resolves the weights file and discovers `tokenizer.json` automatically.

### Out of Scope

**NVIDIA Parakeet TDT** — Parakeet is a FastConformer-TDT (Token-and-Duration Transducer) architecture, fundamentally
different from Whisper's encoder-decoder transformer. It uses an LSTM prediction network, SentencePiece tokenizer, and
transducer decoding — none of which share code with the Whisper pipeline. Supporting Parakeet would require a separate
inference engine.

## Performance

### whisper4j vs Python whisper (203s audio, greedy beam=1, Apple Silicon M-series)

| Model          | Python (PyTorch CPU) | whisper4j (Pure Java) | Ratio |
|----------------|----------------------|-----------------------|-------|
| tiny.en        | 3.8s (53.9x)         | 6.7s (30.2x)          | 0.56x |
| base.en        | 7.4s (27.5x)         | 10.5s (19.3x)         | 0.70x |
| small.en       | 17.0s (12.0x)        | 22.4s (9.1x)          | 0.76x |
| medium.en      | 40.0s (5.1x)         | 84.6s (2.4x)          | 0.47x |
| large-v3-turbo | 28.8s (7.0x)         | 73.3s (2.8x)          | 0.39x |

Python: whisper 20250625, PyTorch 2.11.0, CPU, 8 threads.
Java: whisper4j, Java 26 with Vector API, Apple Accelerate BLAS via Panama FFM.

All models transcribe above real-time. whisper4j reaches 39–76% of Python speed in pure Java with zero native
dependencies. The gap is from PyTorch's fused CUDA-style CPU kernels (batched attention, fused LayerNorm+GELU) that
whisper4j implements as separate BLAS calls.

### Acceleration

- **macOS (Apple Silicon):** Apple Accelerate BLAS via AMX coprocessor (Panama FFM)
- **Java 26+ with `--enable-preview`:** Vector API (JEP 529) for element-wise ops, GELU, softmax
- **Fallback:** Pure Java scalar loops and tiled matrix multiply (Java 25+, or Java 26 without `--enable-preview`)

## Concurrency

`WhisperModel` is thread-safe. Load the model once and call `transcribe()` from multiple threads concurrently.
All model weights are immutable (`final` fields). KV caches and intermediate tensors are allocated per-call.
The encoder's scoped arena uses `ThreadLocal` storage, so each thread gets its own allocation pool.

```java
var model = WhisperModel.load(Path.of("ggml-base.en.bin"));

// Transcribe multiple files concurrently
try (var pool = Executors.newFixedThreadPool(4)) {
    List<Future<List<Segment>>> futures = audioFiles.stream()
        .map(audio -> pool.submit(() -> model.transcribe(audio).toList()))
        .toList();
    for (var f : futures) {
        f.get().forEach(seg -> System.out.println(seg.text()));
    }
}
```

Each concurrent `transcribe()` call:
- Allocates its own KV cache (`HashMap` per decode call)
- Allocates intermediate tensors via `Arena.ofAuto()` or thread-local scoped arena
- Shares only the immutable model weights (read-only `Tensor` segments)

Memory usage scales linearly with concurrent calls — each active transcription uses ~50–500 MB
depending on model size (encoder intermediates + KV cache growth).

## Architecture

```
com.sparrowlogic.whisper4j
├── WhisperModel          # Main entry point: load(), transcribe(), detectLanguage()
├── Alignment             # DTW word timestamp alignment
├── audio/
│   ├── WavParser         # RIFF/WAV parser (PCM 8/16/24/32, float, μ-law)
│   ├── Resampler         # Any rate → 16kHz mono
│   ├── FeatureExtractor  # Log-mel spectrogram (Bluestein FFT)
│   └── VoiceActivityDetector  # Energy-based VAD
├── tokenizer/
│   └── WhisperTokenizer  # BPE tokenizer with special tokens
├── model/
│   ├── ModelLoader       # Auto-detect format
│   ├── GgmlLoader        # GGML binary (mmap, F16/Q4/Q5/Q8 dequant)
│   ├── SafeTensorsLoader # HuggingFace SafeTensors
│   ├── PyTorchLoader     # PyTorch .pt (ZIP + pickle)
│   ├── OnnxLoader        # ONNX protobuf
│   └── CTranslate2Loader # CTranslate2 binary
├── nn/
│   ├── WhisperEncoder    # Audio encoder (Conv1D + transformer)
│   ├── WhisperDecoder    # Text decoder (transformer + KV cache)
│   ├── MultiHeadAttention
│   ├── ResidualAttentionBlock
│   ├── Linear, LayerNorm, Conv1d
└── tensor/
    ├── Tensor            # Off-heap tensor with BLAS acceleration
    ├── SimdOps           # Runtime SIMD detection (scalar fallback or Vector API)
    └── VectorSimdOps     # Vector API impl (multi-release JAR, Java 26+ only)
```

## Building

```bash
# Compile (base Java 25 + Java 26 Vector API overlay)
./mvnw compile

# Run checkstyle
./mvnw checkstyle:check

# Run tests (scalar path — no Vector API)
./mvnw test

# Run tests (SIMD path — with Vector API)
./mvnw test -Pvector-api

# Run transcription test
java --enable-native-access=ALL-UNNAMED \
     -cp target/test-classes:target/classes \
     com.sparrowlogic.whisper4j.TestTranscribeFiles
```

## Requirements

- **Java 25+** (Corretto, Temurin, or Oracle)
- **Java 26+** with `--enable-preview` for Vector API SIMD acceleration (optional)
- **Maven 3.9+** (wrapper included)
- macOS recommended for Accelerate BLAS (works on Linux/Windows at reduced speed)

## License

MIT — see [LICENSE.md](LICENSE.md).

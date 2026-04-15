# Implementation Plan — whisper4j: Pure Java Whisper Inference Engine

## Problem Statement

There is no pure Java implementation of OpenAI's Whisper speech-to-text model. Existing Java solutions rely on native bindings (whisper.cpp via JNI, ONNX Runtime). The goal is a zero-dependency, 100% Java implementation that loads pre-trained Whisper models from any common format, runs inference using Java 26's Vector API for SIMD acceleration, and provides a streaming `Stream<Segment>` API.

## Requirements

- Pure Java 26 — leverage Vector API (JEP 529), Structured Concurrency (JEP 525), Lazy Constants (JEP 526), records, pattern matching (JEP 530), Foreign Memory API (Panama)
- Universal model loading: PyTorch .pt, ONNX .onnx, CTranslate2 model.bin, SafeTensors .safetensors, GGML .bin
- All Whisper model sizes: tiny through large-v3 and turbo
- Audio input: WAV/RIFF via direct ByteBuffer parsing (no javax.sound.sampled), raw float[] PCM arrays, μ-law decoding
- Streaming API: `Stream<Segment>` for speed
- Library-only Maven artifact
- Incorporate faster-whisper efficiency: VAD filtering, batched inference, temperature fallback, KV caching
- Future: CoreML acceleration on Apple Silicon

## Architecture

```
Audio Input → AudioDecoder → FeatureExtractor (log-mel spectrogram)
                                    ↓
                            WhisperEncoder (Conv1D stem + N attention blocks)
                                    ↓
                            WhisperDecoder (token embedding + N cross-attention blocks)
                                    ↓
                            Stream<Segment> (text output)

Model Files → ModelLoader (auto-detect format) → WeightStore → Encoder/Decoder
```

## Reference Implementations

- [openai/whisper](https://github.com/openai/whisper) @ `cba3768` — OpenAI's original Python implementation
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — optimized Python implementation with CTranslate2
- `models/` — GGML model files (whisper.cpp format, source of truth)

## Current Status

### ✅ COMPLETED

#### Task 1: Project Scaffolding and Tensor Core
- Maven project with Java 26, `--enable-preview`, `--add-modules jdk.incubator.vector`
- `module-info.java` with `requires jdk.incubator.vector`, `java.logging`, `java.desktop`
- `Tensor` class backed by off-heap `MemorySegment` (Panama Foreign Memory API)
  - SIMD-accelerated via Vector API: `matmul`, `add`, `scale`, `softmax`, `gelu`, `layerNorm`
  - `FloatVector.fromMemorySegment()` for direct off-heap SIMD operations
  - Shape operations: `reshape`, `slice`, `transpose`
  - Factory methods: `of(float[])`, `ofSegment(MemorySegment)`, `zeros()`, `allocate(Arena)`
- Unit tests for matmul, GELU, softmax, layerNorm, shape operations

#### Task 2: Audio Pipeline
- `WavParser` — RIFF/WAV parser using direct `ByteBuffer` reads (no javax.sound.sampled)
  - Supports PCM 8/16/24/32-bit, IEEE float 32/64-bit, μ-law (format tag 7)
  - μ-law decode ported from pots-voice `UlawCodec` pattern
- `Resampler` — linear interpolation, any rate to 16kHz mono
- `FeatureExtractor` — log-mel spectrogram matching OpenAI/faster-whisper
  - Pure Java FFT (Cooley-Tukey radix-2)
  - Slaney-style mel filter bank
  - Supports 80 and 128 mel bins (for large-v3)
  - `padOrTrim()` to 3000 frames (30s chunks)
- Unit tests for WAV parsing, FFT, mel filters, feature extraction

#### Task 3: Tokenizer
- `WhisperTokenizer` — BPE tokenizer with byte-level fallback
  - Parses HuggingFace `tokenizer.json` format (minimal JSON parser, no dependency)
  - Special tokens: SOT, EOT, timestamps, language (99 languages), task
  - `encode()`, `decode()`, `decodeWithTimestamps()`, `sotSequence()`, `nonSpeechTokens()`
  - Default tokenizer builder from model dimensions (for GGML models without tokenizer.json)
- Unit tests for special token IDs, SOT sequence, encode/decode

#### Task 4: Model Weight Loading — SafeTensors and PyTorch
- `ModelLoader` sealed interface with auto-detection via `forPath()`
- `WeightStore` — name → Tensor map
- `ModelDimensions` record with `infer()` from weight shapes
- `SafeTensorsLoader` — 8-byte header, JSON metadata, mmap via `FileChannel`/`ByteBuffer`
  - F32, F16, BF16 support
- `PyTorchLoader` — ZIP archive + minimal pickle protocol interpreter
  - Handles `torch.save()` format with `model_state_dict` + `dims`
  - FloatStorage, HalfStorage, BFloat16Storage

#### Task 5: Model Weight Loading — ONNX, CTranslate2, and GGML
- `OnnxLoader` — minimal protobuf wire format reader (no protobuf dependency)
  - Parses ModelProto → GraphProto → TensorProto initializers
- `CTranslate2Loader` — binary format parser via `ByteBuffer`
  - Version, model name, revision, variables with name/shape/dtype/bytes, aliases
- `GgmlLoader` — whisper.cpp GGML binary format (**primary loader for testing**)
  - Panama `FileChannel.map()` returning `MemorySegment` (no 2GB limit)
  - F32 tensors: zero-copy mmap slices
  - F16 tensors: dequantized into arena-allocated off-heap memory
  - Quantized types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 dequantization
  - 2D weight matrices: column-major → row-major transposition during load
  - Extracts hyperparams, mel filters, and vocab directly from file
  - OOM-safe: catches `OutOfMemoryError`, cleans up arena on failure
  - Progress logging: tensor count, file percentage, heap usage
  - Memory logging: before/after load, heap vs off-heap tracking
  - Tested successfully with ggml-base.en.bin (147MB), ggml-large-v3.bin (3GB), ggml-large-v3-turbo.bin (1.6GB)

#### Task 6: Whisper Encoder
- `Conv1d` — im2col + SIMD matmul (not naive nested loops)
  - Handles GGML bias shape `[outCh, 1]` → flattened to `[outCh]`
- `MultiHeadAttention` — rearrange to per-head tensors, use Tensor.matmul for QK^T and attn@V
- `ResidualAttentionBlock` — self-attention + optional cross-attention + MLP (GELU)
- `LayerNorm`, `Linear` (PyTorch convention: `x @ W^T + b`)
- `WhisperEncoder` — Conv1D stem + sinusoidal positional embedding + N blocks + ln_post
- Per-block timing logging

#### Task 7: Whisper Decoder and Greedy Decoding
- `WhisperDecoder` — token embedding + learned positional embedding + N cross-attention blocks + ln + logits
- Causal attention mask (upper triangular -inf)
- KV cache for efficient autoregressive generation
- Greedy decoding loop with per-step timing logging

#### Task 9: Streaming Transcription Pipeline and Public API
- `WhisperModel` — main entry point
  - `load(Path)` auto-detects format, builds encoder/decoder/tokenizer
  - `transcribe(Path)` for WAV files
  - `transcribe(float[])` for raw PCM
  - Returns `Stream<Segment>` (lazy)
  - 30-second windowed processing
- `Segment` record (id, start, end, text, tokens)
- `TranscriptionOptions` record (language, task, beamSize, withTimestamps)
- Error handling: clean arena cleanup on load failure

#### Test Harnesses
- `TestMicrophone` — live mic capture via `javax.sound.sampled.TargetDataLine`, 5-second chunks, RMS silence detection
- `TestTranscribeFiles` — file-based transcription using reference test WAVs from faster-whisper
- `GgmlLoaderTest` — validates tensor counts, shapes, dimensions against known models
- `run-test.sh` — shell script for running tests with correct classpath and VM options

#### Task 8: Temperature Fallback and Token Suppression ✅
- Temperature fallback: tries [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Compression ratio check (> 2.4 triggers retry)
- Average log probability check (< -1.0 triggers retry)
- No-speech detection via softmax at SOT position
- Token suppression: SOT, translate, transcribe, startoflm, startofprev, nospeech
- Blank suppression at first generated token

#### Task 10: Language Detection ✅
- `detectLanguage()` via decoder logits at SOT position
- Returns best language from softmax over language tokens

#### Task 11: VAD Filtering ✅
- Energy-based VAD with 30ms windows, RMS threshold
- Configurable min speech/silence duration
- `vadFilter` option in TranscriptionOptions

### 🔧 REMAINING

#### Beam Search (Task 8 enhancement)
- Beam search with configurable beam_size, patience, length_penalty
- Would further reduce repetition in first chunk

#### Word-Level Timestamps (Task 10 enhancement)
- Word timestamps via cross-attention alignment + DTW
- `Word` record (start, end, word, probability)

#### Batched Inference (Task 11 enhancement)
- Structured Concurrency (JEP 525) for parallel chunk processing

#### Future: CoreML / MLX Acceleration
- Java → local Swift service → CoreML (ANE) architecture
- Avoids Obj-C bridging, keeps Java focused on orchestration
- Would dramatically accelerate encoder (~450ms → ~50ms per chunk)

## Performance Observations

### Model Loading (ggml-large-v3-turbo, 1.6GB)
- Load time: **806 ms** (mmap, no heap copy)
- Heap usage: **19 MB** (all weights off-heap via Panama MemorySegment)
- 587 tensors, 3235 MB as f32

### Model Loading (ggml-base.en, 147MB)
- Load time: **239 ms**
- Heap usage: **15 MB**
- 245 tensors, 290 MB as f32

### Encoder Performance (base.en, 6 layers, 512-dim)
- Conv1+GELU: **187 ms** (im2col + SIMD matmul)
- Conv2+GELU: **285 ms**
- Encoder block: **~9s each** (attention matmul bottleneck)
- Total encoder: **~55s** for 30s of audio

### Decoder Performance (base.en, 6 layers, 512-dim)
- First step: **~5.5s** (no KV cache)
- Subsequent steps: **~1s each** (with KV cache)

### Key Bottleneck
The attention mechanism's QK^T matmul `(1500, 64) × (64, 1500)` per head, 8 heads, 6 layers is the dominant cost. The `data()` copy from off-heap MemorySegment to on-heap float[] for the head rearrangement is also expensive. Optimizations needed:
1. Keep head rearrangement in off-heap MemorySegment operations
2. Tile the matmul for better cache locality
3. Consider CoreML offload for the encoder on Apple Silicon

## File Structure

```
src/main/java/
  module-info.java
  com/sparrowlogic/whisper4j/
    WhisperModel.java              — Main entry point, load + transcribe API
    tensor/
      Tensor.java                  — MemorySegment-backed tensor with SIMD ops
    audio/
      WavParser.java               — RIFF/WAV parser (ByteBuffer, no javax.sound)
      Resampler.java               — Sample rate conversion
      FeatureExtractor.java        — Log-mel spectrogram (FFT, mel filters)
    tokenizer/
      WhisperTokenizer.java        — BPE tokenizer with special tokens
    model/
      ModelLoader.java             — Sealed interface, auto-detect format
      WeightStore.java             — Name → Tensor map
      ModelDimensions.java         — Architecture dimensions record
      GgmlLoader.java              — whisper.cpp GGML format (Panama mmap)
      SafeTensorsLoader.java       — HuggingFace SafeTensors format
      PyTorchLoader.java           — PyTorch .pt (ZIP + pickle)
      CTranslate2Loader.java       — CTranslate2 binary format
      OnnxLoader.java              — ONNX protobuf format
    nn/
      Linear.java                  — y = x @ W^T + b
      LayerNorm.java               — Layer normalization
      Conv1d.java                  — 1D convolution (im2col + matmul)
      MultiHeadAttention.java      — Multi-head attention with KV cache
      ResidualAttentionBlock.java  — Attention + MLP block
      WhisperEncoder.java          — Audio encoder
      WhisperDecoder.java          — Text decoder with greedy decode

src/test/java/
  com/sparrowlogic/whisper4j/
    TestMicrophone.java            — Live mic transcription test
    TestTranscribeFiles.java       — File-based transcription test
    tensor/TensorTest.java
    audio/WavParserTest.java
    audio/FeatureExtractorTest.java
    tokenizer/WhisperTokenizerTest.java
    model/GgmlLoaderTest.java
```

## Dependencies

- **Runtime**: None (pure Java 26 + incubator Vector API)
- **Test**: JUnit 5.12.1

## Java 26 Features Used

| Feature | JEP | Usage |
|---------|-----|-------|
| Vector API | 529 | SIMD matmul, dot product, add, scale in Tensor |
| Foreign Memory API | (finalized) | Off-heap model weights via MemorySegment + Arena |
| Records | (finalized) | Segment, TranscriptionOptions, ModelDimensions, WavData, TensorMeta |
| Pattern Matching | 530 | `instanceof GgmlLoader ggml` in WhisperModel.load() |
| Sealed Classes | (finalized) | `ModelLoader` sealed interface |
| Text Blocks | (finalized) | Test tokenizer JSON |
| Switch Expressions | (finalized) | Dtype dispatch in loaders, audio format dispatch |
| Structured Concurrency | 525 | Planned for batched inference (Task 11) |
| Lazy Constants | 526 | Planned for deferred model loading (Task 9 enhancement) |

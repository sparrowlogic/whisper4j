# Benchmark: JDK 26 vs Valhalla vs GraalVM Native Image

Comparing whisper4j performance across Corretto 26, OpenJDK Valhalla, and GraalVM native-image,
with and without Vector API preview features.

**Date:** 2026-04-16

---

## Versions

| Component        | Version / Commit                                              |
|------------------|---------------------------------------------------------------|
| whisper4j        | `9c3f31abca0866dc06826e147cf683b4617225c2`                    |
| Corretto 26      | `26+35-FR` (Amazon Corretto-26.0.0.35.1)                      |
| OpenJDK Valhalla | `8f9586645baae882b8a60e3111d00c600cdf1089` (branch: `lworld`) |
| GraalVM Oracle   | `26-dev+13.1` (26.ea.13-graal, Substrate VM)                  |
| Maven            | 3.9.11 (wrapper)                                              |

## Hardware

|     |                             |
|-----|-----------------------------|
| CPU | Apple M2 Max                |
| RAM | 96 GB                       |
| OS  | macOS 26.4.1 (Build 25E253) |

---

## Prerequisites

### 1. Build OpenJDK Valhalla from source

```bash
git clone https://github.com/openjdk/valhalla.git openjdk-valhalla
cd openjdk-valhalla
git checkout 8f9586645baae882b8a60e3111d00c600cdf1089

bash configure
make images

# Verify
./build/macosx-aarch64-server-release/jdk/bin/java -version
# openjdk version "27-internal" 2026-09-15
```

### 2. Install Corretto 26

```bash
sdk install java 26-amzn
sdk use java 26-amzn
```

### 3. Install GraalVM Oracle 26 EA

```bash
sdk install java 26.ea.13-graal
```

### 4. Clone whisper4j at the tested commit

```bash
git clone https://github.com/sparrowlogic/whisper4j.git
cd whisper4j
git checkout 9c3f31abca0866dc06826e147cf683b4617225c2
```

---

## Reproduction Commands

```bash
export VALHALLA_HOME=/path/to/openjdk-valhalla/build/macosx-aarch64-server-release/jdk
export GRAALVM_HOME=$HOME/.sdkman/candidates/java/26.ea.13-graal
```

### Unit test suite (4 JVM configurations)

```bash
# Corretto 26 — no preview
./mvnw test -B 2>&1 | tee /tmp/whisper4j-jdk26-nopreview.log

# Corretto 26 — with Vector API
./mvnw test -Pvector-api -B 2>&1 | tee /tmp/whisper4j-jdk26-preview.log

# Valhalla — no preview
JAVA_HOME=$VALHALLA_HOME ./mvnw test -B 2>&1 | tee /tmp/whisper4j-valhalla-nopreview.log

# Valhalla — with Vector API
JAVA_HOME=$VALHALLA_HOME ./mvnw test -Pvector-api -B 2>&1 | tee /tmp/whisper4j-valhalla-preview.log
```

### Transcription benchmark (5 platforms)

```bash
# Compile
./mvnw compile -B

MODEL=models/ggml-base.en.bin
AUDIO=src/test/resources/data/physicsworks.wav
CP=target/classes
MOD=com.sparrowlogic.whisper4j/com.sparrowlogic.whisper4j.WhisperCli

# Corretto 26 — no preview
java -XX:+UnlockDiagnosticVMOptions -XX:-UseSuperWord \
  --enable-native-access=ALL-UNNAMED \
  -p $CP -m $MOD $MODEL $AUDIO 1

# Corretto 26 — with Vector API
java --enable-preview --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -p "$CP:$CP/META-INF/versions/26" -m $MOD $MODEL $AUDIO 1

# Valhalla — no preview
$VALHALLA_HOME/bin/java -XX:+UnlockDiagnosticVMOptions -XX:-UseSuperWord \
  --enable-native-access=ALL-UNNAMED \
  -p $CP -m $MOD $MODEL $AUDIO 1

# Valhalla — with Vector API
$VALHALLA_HOME/bin/java --enable-preview --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -p "$CP:$CP/META-INF/versions/26" -m $MOD $MODEL $AUDIO 1

# GraalVM native-image
JAVA_HOME=$GRAALVM_HOME ./mvnw package -Pnative -DskipTests -B
./target/whisper4j $MODEL $AUDIO 1
```

---

## Results: Transcription Benchmark

physicsworks.wav (203s audio), base.en model, beam=1, single run.

| Platform                 | Load (ms) | Transcribe (ms) | Realtime Factor | vs Baseline |
|--------------------------|-----------|-----------------|-----------------|-------------|
| Corretto 26 (no preview) | 502       | 11,571          | 17.6x           | — baseline — |
| Corretto 26 + Vector API | 452       | 11,824          | 17.2x           | ~same       |
| Valhalla (no preview)    | 576       | 10,410          | 19.5x           | +10.0%      |
| Valhalla + Vector API    | 486       | 10,441          | 19.5x           | +10.0%      |
| GraalVM native-image     | 11,853    | >1,800,000†     | <0.1x           | >150x slower |

†Killed after 30+ minutes of CPU time. Did not complete transcription.

## Results: Unit Test Suite

131 tests, physicsworks.wav transcription + validation + concurrency.

| Configuration            | Tests | Failures | Total Time | vs Baseline  |
|--------------------------|-------|----------|------------|--------------|
| Corretto 26 (no preview) | 131   | 0        | 3:24       | — baseline — |
| Corretto 26 + Vector API | 131   | 0        | 2:56       | -13.7%       |
| Valhalla (no preview)    | 131   | 2†       | 3:06       | -8.8%        |
| Valhalla + Vector API    | 131   | 0        | 2:58       | -12.7%       |

†2 failures in `SimdOpsTest` — vector API detection tests only. All computation tests passed.

### Per-Test Timing (seconds)

| Test Class                   | Corretto | Corretto+Vec | Valhalla | Valhalla+Vec |
|------------------------------|----------|--------------|----------|--------------|
| ValidateAgainstReferenceTest | 190.3    | 163.4        | 172.2    | 165.7        |
| ConcurrencyTest              | 11.80    | 10.62        | 10.67    | 10.54        |
| WhisperModelFactoryTest      | 1.155    | 0.808        | 1.106    | 0.862        |
| FeatureExtractorTest         | 0.245    | 0.242        | 0.245    | 0.243        |
| GgmlLoaderTest               | 0.188    | 0.057        | 0.181    | 0.065        |

### Encoder Chunk Performance (203s audio, base.en)

| Chunk   | Corretto   | Corretto+Vec | Valhalla   | Valhalla+Vec |
|---------|------------|--------------|------------|--------------|
| 0.0s    | 652ms      | 343ms        | 379ms      | 348ms        |
| 47.9s   | 642ms      | 333ms        | 409ms      | 316ms        |
| 111.9s  | 697ms      | 292ms        | 421ms      | 307ms        |
| **Avg** | **~664ms** | **~323ms**   | **~403ms** | **~324ms**   |

---

## GraalVM Native Image Details

| Metric          | Value                                                |
|-----------------|------------------------------------------------------|
| Binary size     | 18 MB                                                |
| Build time      | 29.6s                                                |
| Build flags     | `-H:+ForeignAPISupport -H:+SharedArenaSupport -march=native --no-fallback` |
| Vector API      | Not available (jdk.incubator.vector unsupported in Substrate VM) |
| BLAS            | Apple Accelerate via Panama FFM — **requires `reachability-metadata.json`** |
| Model load      | 11,853ms (vs ~500ms on JVM) — no JIT for F16→F32 dequant loops |
| Transcription   | >30 min for 203s audio (killed) — ~150x slower than JVM |

### Critical: FFM Downcall Registration

The initial native-image build appeared to work but **silently failed to load Apple Accelerate BLAS**.
The `AccelerateBlas` static initializer catches all exceptions, so the downcall handle creation
failed without error and fell back to pure scalar matmul — making the binary ~150x slower.

Per the [GraalVM FFM API docs](https://www.graalvm.org/jdk25/reference-manual/native-image/native-code-interoperability/ffm-api/),
native-image requires explicit downcall descriptors registered at build time via `reachability-metadata.json`.
After adding the proper descriptors for `cblas_sgemm`, `vvexpf`, `vDSP_maxv`, `vDSP_vsadd`, `vDSP_sve`,
`vDSP_vsdiv`, `vDSP_svdiv`, `vDSP_vsmul`, and `vDSP_vmul`, Accelerate loads successfully.

With BLAS working, the encoder runs at ~3.7s/chunk (vs ~0.6s on JVM — 6x slower).
The decoder remains the bottleneck at ~49s/chunk (vs ~4.6s on JVM — 10x slower)
because the autoregressive loop is pure Java scalar code that depends on JIT optimization.

---

## Analysis

- **Valhalla is 10% faster** than Corretto 26 for end-to-end transcription (10.4s vs 11.6s).
  The improvement comes from better C2 JIT compilation of scalar loops.
- **Vector API has minimal impact on transcription** for base.en — Apple Accelerate BLAS
  dominates the matmul cost, so the Vector API's element-wise speedup is marginal.
  The Vector API benefit is more visible in the unit test suite (encoder chunks: 664ms → 323ms)
  because the test suite exercises the encoder more heavily relative to BLAS.
- **Valhalla + Vector API ≈ Valhalla alone** for transcription. The BLAS path is the bottleneck.
- **GraalVM native-image is not viable** for this workload. Without JIT, the autoregressive
  decoder loop (hundreds of sequential matmul + softmax + attention steps per chunk) runs
  >150x slower. The AOT compiler cannot match the JIT's speculative optimizations for
  hot loops with complex control flow. PGO (`--pgo`) might help but was not tested.
- **Model loading is 24x slower** in native-image (11.8s vs 0.5s) because the F16→F32
  dequantization loops don't benefit from JIT warmup.

---

## Notes

- The `vector-api` Maven profile adds `--enable-preview --add-modules jdk.incubator.vector`
  to surefire and puts the multi-release Java 26 classes on the classpath.
- The default surefire config includes `-XX:+UnlockDiagnosticVMOptions -XX:-UseSuperWord`
  as a workaround for a Corretto 26 EA SIGSEGV. Harmless on Valhalla.
- Native-image requires `-H:+SharedArenaSupport` for `Arena.ofShared()` used in the encoder.
- Native-image requires `-H:+ForeignAPISupport` for Panama FFM downcall handles (Accelerate BLAS).
- The `native` Maven profile builds the binary: `./mvnw package -Pnative -DskipTests`

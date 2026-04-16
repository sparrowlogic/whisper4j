# Benchmark: JDK 26 vs Valhalla (Value Types)

Comparing whisper4j unit test performance across Corretto 26 and OpenJDK Valhalla,
with and without Vector API preview features.

**Date:** 2026-04-16

---

## Versions

| Component        | Version / Commit                                              |
|------------------|---------------------------------------------------------------|
| whisper4j        | `9c3f31abca0866dc06826e147cf683b4617225c2`                    |
| Corretto 26      | `26+35-FR` (Amazon Corretto-26.0.0.35.1)                      |
| OpenJDK Valhalla | `8f9586645baae882b8a60e3111d00c600cdf1089` (branch: `lworld`) |
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
# OpenJDK Runtime Environment (build 27-internal-adhoc....)
# OpenJDK 64-Bit Server VM (build 27-internal-adhoc...., mixed mode)
```

On Linux the build output path will be `build/linux-x86_64-server-release/jdk/bin/java` (or similar).

### 2. Install Corretto 26

```bash
# Via SDKMAN
sdk install java 26-amzn
sdk use java 26-amzn

java -version
# openjdk version "26" 2026-03-17
# OpenJDK Runtime Environment Corretto-26.0.0.35.1 (build 26+35-FR)
```

### 3. Clone whisper4j at the tested commit

```bash
git clone https://github.com/sparrowlogic/whisper4j.git
cd whisper4j
git checkout 9c3f31abca0866dc06826e147cf683b4617225c2
```

---

## Reproduction Commands

Set `VALHALLA_HOME` to your Valhalla build output:

```bash
export VALHALLA_HOME=/path/to/openjdk-valhalla/build/macosx-aarch64-server-release/jdk
```

### Run 1: Corretto 26 — no preview

```bash
./mvnw test -B 2>&1 | tee /tmp/whisper4j-jdk26-nopreview.log
```

### Run 2: Corretto 26 — with Vector API

```bash
./mvnw test -Pvector-api -B 2>&1 | tee /tmp/whisper4j-jdk26-preview.log
```

### Run 3: Valhalla — no preview

```bash
JAVA_HOME=$VALHALLA_HOME ./mvnw test -B 2>&1 | tee /tmp/whisper4j-valhalla-nopreview.log
```

### Run 4: Valhalla — with Vector API

```bash
JAVA_HOME=$VALHALLA_HOME ./mvnw test -Pvector-api -B 2>&1 | tee /tmp/whisper4j-valhalla-preview.log
```

### Extract results

```bash
for f in /tmp/whisper4j-*.log; do
  echo "=== $(basename $f) ==="
  grep -E "Tests run:.*Time elapsed" "$f"
  grep "Total time" "$f"
  echo
done
```

---

## Results

### Overall Test Suite

| Configuration            | Tests | Failures | Total Time | vs Baseline  |
|--------------------------|-------|----------|------------|--------------|
| Corretto 26 (no preview) | 131   | 0        | 3:24       | — baseline — |
| Corretto 26 + Vector API | 131   | 0        | 2:56       | -13.7%       |
| Valhalla (no preview)    | 131   | 2†       | 3:06       | -8.8%        |
| Valhalla + Vector API    | 131   | 0        | 2:58       | -12.7%       |

†2 failures in `SimdOpsTest` — vector API detection tests only. Valhalla without `--enable-preview`
does not expose `jdk.incubator.vector` the same way. All computation tests passed.

### Per-Test Timing (seconds)

| Test Class                   | Corretto | Corretto+Vec | Valhalla | Valhalla+Vec |
|------------------------------|----------|--------------|----------|--------------|
| ValidateAgainstReferenceTest | 190.3    | 163.4        | 172.2    | 165.7        |
| ConcurrencyTest              | 11.80    | 10.62        | 10.67    | 10.54        |
| WhisperModelFactoryTest      | 1.155    | 0.808        | 1.106    | 0.862        |
| FeatureExtractorTest         | 0.245    | 0.242        | 0.245    | 0.243        |
| GgmlLoaderTest               | 0.188    | 0.057        | 0.181    | 0.065        |
| TensorTest                   | 0.010    | 0.012        | 0.011    | 0.010        |
| SimdOpsTest                  | 0.003    | 0.002        | 0.009    | 0.001        |

### Encoder Chunk Performance (203s audio, base.en model)

| Chunk   | Corretto   | Corretto+Vec | Valhalla   | Valhalla+Vec |
|---------|------------|--------------|------------|--------------|
| 0.0s    | 652ms      | 343ms        | 379ms      | 348ms        |
| 47.9s   | 642ms      | 333ms        | 409ms      | 316ms        |
| 111.9s  | 697ms      | 292ms        | 421ms      | 307ms        |
| **Avg** | **~664ms** | **~323ms**   | **~403ms** | **~324ms**   |

### Peak Heap (MB)

| Corretto | Corretto+Vec | Valhalla | Valhalla+Vec |
|----------|--------------|----------|--------------|
| 354      | 384          | 351      | 365          |

---

## Analysis

- **Valhalla scalar path is 39% faster** for encoder chunks vs Corretto without preview
  (664ms → 403ms avg). This comes from improved C2 JIT compilation of scalar loops
  (element-wise ops, GELU, softmax, tiled matmul).
- **With Vector API, both JDKs converge** (~323ms vs ~324ms). SIMD intrinsics replace
  the scalar loops that Valhalla optimizes, eliminating the advantage.
- **Memory is comparable** across all configurations (351–384 MB peak heap).
- **If you cannot use `--enable-preview`**, Valhalla gives ~40% of the Vector API speedup
  for free through better JIT alone.

---

## Notes

- The `vector-api` Maven profile adds `--enable-preview --add-modules jdk.incubator.vector`
  to surefire and puts the multi-release Java 26 classes on the classpath.
- The default surefire config includes `-XX:+UnlockDiagnosticVMOptions -XX:-UseSuperWord`
  as a workaround for a Corretto 26 EA SIGSEGV. This flag is harmless on Valhalla.
- Valhalla's `jdk.incubator.vector` module availability differs from standard JDK 26.
  The 2 `SimdOpsTest` failures in the no-preview run are detection tests, not correctness tests.

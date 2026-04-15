# Apple Silicon Hardware Acceleration for whisper4j

## Overview

Apple Silicon (M1–M5) provides four distinct compute paths for ML workloads, each with different performance characteristics and Java accessibility:

| Compute Unit | Peak GFLOPS (f32, single core) | Java Access Path | Effort |
|---|---|---|---|
| ARM NEON (128-bit SIMD) | ~102 | Java Vector API (direct) | Already done |
| AMX (Matrix Coprocessor) | ~1,475 | FFM API → Accelerate/vecLib | Medium |
| Apple Neural Engine (ANE) | Varies by chip | FFM API → Core ML (Obj-C bridge) | High |
| Metal GPU | Varies by chip | FFM API → Metal (Obj-C bridge) | High |

The key insight: AMX delivers roughly 14x the throughput of NEON for matrix multiply operations, and Apple's Accelerate framework (BLAS/vecLib) uses AMX internally. We can call Accelerate from Java via the Foreign Function & Memory (FFM) API with zero native code.

---

## Tier 1: Java Vector API → NEON (Current)

### What We Have Today

whisper4j already uses `jdk.incubator.vector` (JEP 529, 11th incubation in Java 26). On Apple Silicon, `FloatVector.SPECIES_PREFERRED` resolves to `SPECIES_128` (4 float lanes), which the HotSpot C2 compiler maps to ARM NEON instructions.

This works well for element-wise operations:
- `Tensor.add()`, `Tensor.scale()` — vectorized add/multiply
- `Tensor.gelu()`, `Tensor.softmax()`, `Tensor.layerNorm()` — loop-level SIMD
- `Tensor.simdDot()` — dot product via `FloatVector.mul().reduceLanes(ADD)`

### Limitations

NEON is 128-bit only (no SVE on Apple Silicon as of M5). For matrix multiply, NEON tops out around 102 GFLOPS on a single core — the bottleneck for `Tensor.matmul()`.

The Vector API uses SLEEF (SIMD Library for Evaluating Elementary Functions) on ARM/macOS for transcendental operations (`exp`, `log`, `tanh`, etc.), linked via FFM internally.

### No Action Required

This tier is already implemented. The Vector API will eventually leave incubation once Project Valhalla value classes are available.

---

## Tier 2: FFM API → Apple Accelerate (BLAS/vDSP) — Recommended Next Step

### Why This Matters

Apple's Accelerate framework (`/System/Library/Frameworks/Accelerate.framework`) provides BLAS, LAPACK, and vDSP routines that internally dispatch to the AMX coprocessor for matrix operations. This is the single highest-impact optimization available:

- `cblas_sgemm` (matrix multiply): ~1,400+ GFLOPS via AMX vs ~102 via NEON — **~14x speedup**
- `vDSP_fft_zrop` (FFT): hardware-optimized, relevant for `FeatureExtractor`
- `vDSP_meanv`, `vDSP_measqv` (statistics): useful for layer normalization

### AMX Background

AMX (Apple Matrix Coprocessor) is an undocumented coprocessor on Apple Silicon. It has:
- 8 X registers and 8 Y registers (64 bytes each) for input operands
- 64 Z registers (64 bytes each) for accumulation
- Outer-product FMA that computes a 16×16 f32 tile per instruction
- 4 parallel ALU groups achieving ~1,666 GFLOPS peak on M2 Pro

AMX is not directly accessible via any public API. Apple exposes it only through Accelerate/vecLib. Calling `cblas_sgemm` is the supported way to use AMX from any language.

### Implementation Plan

#### Step 1: Define CBLAS Bindings via FFM

```java
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Java bindings to Apple Accelerate BLAS via FFM API.
 * Zero native code — pure Java using Panama Foreign Function API.
 */
public final class AccelerateBlas {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup ACCELERATE = SymbolLookup.libraryLookup(
            "/System/Library/Frameworks/Accelerate.framework/Accelerate",
            Arena.global());

    // CblasRowMajor=101, CblasNoTrans=111
    private static final int ROW_MAJOR = 101;
    private static final int NO_TRANS = 111;
    private static final int TRANS = 112;

    // void cblas_sgemm(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    //                  int M, int N, int K,
    //                  float alpha, const float *A, int lda,
    //                  const float *B, int ldb,
    //                  float beta, float *C, int ldc)
    private static final MethodHandle SGEMM = LINKER.downcallHandle(
            ACCELERATE.find("cblas_sgemm").orElseThrow(),
            FunctionDescriptor.ofVoid(
                    ValueLayout.JAVA_INT,   // Order
                    ValueLayout.JAVA_INT,   // TransA
                    ValueLayout.JAVA_INT,   // TransB
                    ValueLayout.JAVA_INT,   // M
                    ValueLayout.JAVA_INT,   // N
                    ValueLayout.JAVA_INT,   // K
                    ValueLayout.JAVA_FLOAT, // alpha
                    ValueLayout.ADDRESS,    // A
                    ValueLayout.JAVA_INT,   // lda
                    ValueLayout.ADDRESS,    // B
                    ValueLayout.JAVA_INT,   // ldb
                    ValueLayout.JAVA_FLOAT, // beta
                    ValueLayout.ADDRESS,    // C
                    ValueLayout.JAVA_INT    // ldc
            ));

    /**
     * C = alpha * A @ B + beta * C
     * A: (M, K), B: (K, N), C: (M, N) — all row-major MemorySegments.
     */
    public static void sgemm(int M, int N, int K,
                             float alpha, MemorySegment A, int lda,
                             MemorySegment B, int ldb,
                             float beta, MemorySegment C, int ldc) {
        try {
            SGEMM.invokeExact(ROW_MAJOR, NO_TRANS, NO_TRANS,
                    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } catch (Throwable t) {
            throw new RuntimeException("cblas_sgemm failed", t);
        }
    }
}
```

#### Step 2: Integrate into Tensor.matmul()

```java
// In Tensor.java — replace the pure-Java matmul with Accelerate dispatch
public Tensor matmul(Tensor b) {
    int M = this.shape[this.shape.length - 2];
    int K = this.shape[this.shape.length - 1];
    int N = b.shape[b.shape.length - 1];
    int batch = this.size / (M * K);
    Tensor out = Tensor.zeros(matmulShape(M, N));

    for (int bi = 0; bi < batch; bi++) {
        long aOff = (long) bi * M * K * Float.BYTES;
        long bOff = (long) bi * K * N * Float.BYTES;
        long cOff = (long) bi * M * N * Float.BYTES;

        AccelerateBlas.sgemm(M, N, K,
                1.0f, this.segment.asSlice(aOff), K,
                b.segment.asSlice(bOff), N,
                0.0f, out.segment.asSlice(cOff), N);
    }
    return out;
}
```

#### Step 3: vDSP for FFT (FeatureExtractor)

```java
// vDSP_fft_zrop — real-to-complex FFT, much faster than our Cooley-Tukey
private static final MethodHandle VDSP_CREATE_FFT = LINKER.downcallHandle(
        ACCELERATE.find("vDSP_create_fftsetup").orElseThrow(),
        FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT,   // log2n
                ValueLayout.JAVA_INT)); // radix (kFFTRadix2 = 0)

private static final MethodHandle VDSP_FFT_ZROP = LINKER.downcallHandle(
        ACCELERATE.find("vDSP_fft_zrop").orElseThrow(),
        FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS,    // setup
                ValueLayout.ADDRESS,    // input (DSPSplitComplex*)
                ValueLayout.JAVA_INT,   // input stride
                ValueLayout.ADDRESS,    // output (DSPSplitComplex*)
                ValueLayout.JAVA_INT,   // output stride
                ValueLayout.JAVA_INT,   // log2n
                ValueLayout.JAVA_INT)); // direction (kFFTDirection_Forward = 1)
```

### Platform Detection

```java
public static boolean isAccelerateAvailable() {
    try {
        SymbolLookup.libraryLookup(
                "/System/Library/Frameworks/Accelerate.framework/Accelerate",
                Arena.ofAuto());
        return true;
    } catch (IllegalArgumentException e) {
        return false;
    }
}
```

The `Tensor.matmul()` implementation should fall back to the pure-Java SIMD path on non-macOS platforms.

### jextract Alternative

Instead of hand-writing FFM bindings, `jextract` can auto-generate them from Accelerate headers:

```bash
jextract \
  --include-function cblas_sgemm \
  --include-function cblas_sscal \
  --include-function vDSP_fft_zrop \
  --include-function vDSP_create_fftsetup \
  -t com.sparrowlogic.whisper4j.native.accelerate \
  -l /System/Library/Frameworks/Accelerate.framework/Accelerate \
  /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Headers/Accelerate.h
```

This generates type-safe Java classes with proper memory layout handling. However, hand-written bindings for the few functions we need are simpler and avoid the jextract build dependency.

### Expected Impact

| Operation | Current (NEON) | With Accelerate (AMX) | Speedup |
|---|---|---|---|
| matmul (encoder forward) | ~100 GFLOPS | ~1,400 GFLOPS | ~14x |
| FFT (feature extraction) | Custom Java | vDSP hardware-optimized | ~5-10x est. |
| Overall transcription | Baseline | Estimated | ~5-8x est. |

Matrix multiply dominates Whisper inference time (attention QKV, linear projections, MLP layers), so the AMX speedup on matmul translates to a large end-to-end improvement.

### Module System Considerations

The FFM API requires `--enable-native-access` at runtime. In `module-info.java`:

```java
module com.sparrowlogic.whisper4j {
    requires jdk.incubator.vector;
    // No additional module requirement for FFM — it's in java.base since JDK 22
}
```

Launch with: `--enable-native-access=com.sparrowlogic.whisper4j`

---

## Tier 3: Core ML / Apple Neural Engine (Stretch Goal)

### What It Provides

The Apple Neural Engine (ANE) is a dedicated NPU present on all Apple Silicon. On M4, it delivers up to 38 TOPS. Core ML is Apple's framework for deploying ML models to ANE, GPU, or CPU — it automatically selects the optimal compute unit.

Apple's research shows ANE-optimized transformers can achieve up to 10x faster inference and 14x lower peak memory compared to baseline implementations.

### Challenges for Java Integration

Core ML is an Objective-C/Swift framework. Calling it from Java requires:

1. Bridging the Objective-C runtime via FFM (`objc_msgSend`, class/selector lookup)
2. Converting Whisper model weights to Core ML format (`.mlmodelc`)
3. Managing `MLMultiArray` ↔ `Tensor` conversions
4. Handling the asynchronous prediction API

This is significantly more complex than calling C functions in Accelerate.

### Possible Approach

```java
// Conceptual — Objective-C runtime bridging via FFM
SymbolLookup objcRuntime = SymbolLookup.libraryLookup("libobjc.dylib", Arena.global());
MethodHandle objc_getClass = LINKER.downcallHandle(
        objcRuntime.find("objc_getClass").orElseThrow(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
MethodHandle sel_registerName = LINKER.downcallHandle(
        objcRuntime.find("sel_registerName").orElseThrow(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
MethodHandle objc_msgSend = LINKER.downcallHandle(
        objcRuntime.find("objc_msgSend").orElseThrow(),
        /* variadic — descriptor depends on the message being sent */);

// Load a compiled Core ML model
// MLModel *model = [MLModel modelWithContentsOfURL:url error:&error];
```

### Recommendation

Defer Core ML/ANE integration. The complexity is high and the Accelerate/AMX path (Tier 2) provides the majority of the performance benefit for matrix-heavy workloads. Revisit if/when a Java-to-Objective-C bridge library matures, or if Apple provides a C API for Core ML.

---

## Tier 4: Metal GPU Compute (Future)

### What It Provides

Metal is Apple's GPU compute API. Metal Performance Shaders (MPS) provide optimized GPU kernels for ML operations. Apple's MLX framework (used by llama.cpp, whisper.cpp on Mac) uses Metal extensively.

### Challenges

Same Objective-C bridging challenges as Core ML, plus:
- Metal shader compilation pipeline
- GPU memory management (shared memory on Apple Silicon simplifies this)
- Synchronization between CPU and GPU work

### Recommendation

Not recommended for whisper4j in the near term. The Accelerate BLAS path already leverages AMX which shares the unified memory architecture. Metal would primarily benefit batch processing or very large models where GPU parallelism outweighs the dispatch overhead.

---

## Implementation Priority

1. **Now**: Continue using Java Vector API for element-wise ops (already done)
2. **Next**: Add FFM → Accelerate `cblas_sgemm` for `Tensor.matmul()` — highest ROI
3. **Then**: Add FFM → vDSP for FFT in `FeatureExtractor` — moderate ROI
4. **Later**: Evaluate Core ML/ANE if a Java Obj-C bridge becomes practical
5. **Maybe**: Metal GPU compute for batch transcription scenarios

## Build & Runtime Requirements

- macOS 12+ (Monterey) for Accelerate with AMX support
- Java 22+ for stable FFM API (`java.lang.foreign`)
- Java 26 for Vector API (incubating, `jdk.incubator.vector`)
- `--enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED`
- No native compilation step — pure Java via FFM

## References

- [JEP 529: Vector API (Eleventh Incubator)](https://openjdk.java.net/jeps/529)
- [JEP 454: Foreign Function & Memory API](https://openjdk.java.net/jeps/454)
- [Apple Accelerate Framework](https://developer.apple.com/accelerate/)
- [AMX Performance Analysis (Zheng's Notes)](https://zhen8838.github.io/posts/mac-amx_en.html)
- [AMX Reverse Engineering (corsix/amx)](https://github.com/corsix/amx)
- [Apple ANE Transformer Optimization](https://machinelearning.apple.com/research/neural-engine-transformers)
- [jextract Tool](https://jdk.java.net/jextract/)
- [Deploying Transformers on Apple Neural Engine (Apple ML Research)](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Characterizing Apple's Neural Engine for LLM Workloads (arXiv)](https://arxiv.org/html/2603.06728v1)

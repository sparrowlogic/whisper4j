# Optimization Notes — whisper4j

Lessons learned during performance tuning. Reference before attempting new optimizations.

## ✅ Optimizations That Worked

### F16 GELU Lookup Table (6x speedup on Conv+GELU)

- Pre-compute GELU for all 65536 F16 values at class load time
- `Float.floatToFloat16(x)` → table index → result
- Matches whisper.cpp's `ggml_table_gelu_f16` approach
- Eliminates `Math.tanh()` per element (the dominant cost)
- **Conv1+GELU: 24ms → 4ms**

### Pre-transposed Linear Weights (2x encoder block speedup)

- Transpose weight matrices once at construction time into native memory
- Forward pass uses `matmul(weightT)` instead of `matmulTransB(weight)`
- Eliminates CblasTrans flag overhead and enables better memory access
- Trade-off: model load time increases (232ms → 478ms) — acceptable one-time cost
- **Encoder block: 80ms → 33ms**

### Native Tensor Allocation (eliminates ensureNative copies)

- `Tensor.ofNative()` allocates off-heap from the start
- Conv1d im2col and bias results use native allocation
- Avoids `ensureNative()` heap→off-heap copy in every matmul call
- Critical for BLAS path which requires native MemorySegment

### Bluestein FFT (23x mel spectrogram speedup)

- Direct DFT for n=400 was O(n²) per frame × 3001 frames = dominant bottleneck
- Bluestein's algorithm converts to power-of-2 FFT via chirp-z convolution
- **Mel spectrogram: 4700ms → 200ms**

### Apple Accelerate BLAS via Panama FFM

- `cblas_sgemm` dispatches to AMX coprocessor on Apple Silicon
- ~14x throughput over pure Java for large matrices
- `vDSP` functions for parallel softmax (softmaxRows)
- `vvexpf` for vectorized exp in softmax
- Key: use `CblasRowMajor` with correct `lda`/`ldb`/`ldc` parameters

### GGML Mel Filters (correctness fix that also improved perf)

- Using pre-computed mel filters from GGML file instead of computing our own
- Eliminates mel filter bank computation and ensures exact match with whisper.cpp

## ❌ Optimizations That Did NOT Work

### SIMD matmul-vec for M=1 on Apple Silicon

- **Attempted:** Replace `cblas_sgemm` with Java Vector API dot products for single-token decoder steps
- **Result:** 2x SLOWER (10.8s → 22.6s for physicsworks.wav)
- **Why:** Apple Accelerate's sgemm already uses AMX coprocessor which is faster than NEON SIMD for any matrix size. The
  BLAS call overhead (~5μs) is real but AMX throughput compensates.
- **Lesson:** On Apple Silicon, ALWAYS use Accelerate BLAS. Don't try to beat AMX with Vector API.
- **Exception:** On non-Apple platforms (no BLAS), the SIMD path IS faster than tiled matmul for M=1.

### SIMD matmul-vec with K threshold (K >= 256)

- **Attempted:** Only use SIMD for large K (Linear layers K=512) but keep BLAS for small K (attention K=64)
- **Result:** Still slower (15.6s vs 10.8s baseline)
- **Why:** The accumulation pattern `out += x[k] * B_row[k]` has poor cache behavior — reads entire output vector N
  times. For N=2048 (MLP), this thrashes L1 cache.
- **Lesson:** Row-accumulation matmul-vec is cache-hostile for large N. Dot-product-per-output-element is better but
  requires column access (stride-N), which is also cache-hostile.

### Pre-allocated KV Cache Buffer

- **Attempted:** Pre-allocate (batchHeads, 448, headDim) buffer, append in-place to avoid O(N²) copying
- **Result:** Introduced bugs — `viewRows` still required copying to create contiguous tensors for BLAS. The `.cap`
  metadata tracking added complexity and broke when caches were shared across decode calls.
- **Why:** BLAS requires contiguous memory per batch element. A pre-allocated buffer with maxLen stride can't be used
  directly — you need to copy the active portion anyway.
- **Lesson:** KV cache append-and-copy is O(N) per step, O(N²) total. For N≤224 tokens, this is ~50K floats copied per
  step — negligible vs the matmul compute. Don't optimize what isn't the bottleneck.

### Sigmoid GELU Approximation (x * σ(1.702x))

- **Attempted:** Use `x / (1 + exp(-1.702x))` via vDSP vectorized exp
- **Result:** Produced wrong transcription output
- **Why:** Whisper was trained with tanh GELU `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`. The sigmoid approximation has ~
  1% relative error which compounds across 6 encoder + 6 decoder blocks.
- **Lesson:** Activation function must match training exactly. Use F16 lookup table for speed without sacrificing
  accuracy.

### Accelerate vDSP GELU (x * sigmoid(1.702x) via vvexpf)

- **Attempted:** Vectorized sigmoid GELU using Apple vDSP functions
- **Result:** Same as above — wrong activation function
- **Lesson:** Same as above. The vDSP approach would work IF we used the tanh formula, but `vvtanhf` doesn't exist in
  Accelerate. The F16 table is simpler and faster.

## 🔶 Partially Effective Optimizations

### matmulTransB (CblasTrans flag)

- Using `sgemmTransB` avoids explicit transpose allocation
- Saves ~0.5ms per encoder block (transpose of 512×512 weight)
- But with pre-transposed weights, this is no longer needed — `matmul` with pre-transposed weight is equivalent and
  slightly faster
- **Current state:** Linear uses pre-transposed weights + `matmul`. `matmulTransB` still used for QK^T in attention (
  where K is not pre-transposed).

### Parallel Softmax (vDSP multi-threaded)

- Splits softmax rows across CPU cores using platform threads
- Helps for large attention matrices (1500×1500 in encoder)
- Negligible for decoder (8×1×N softmax — too small to parallelize)

## 📊 Where Time Is Actually Spent (base.en, single token step)

Per decoder block (2.2ms total):

- Cross-attention: 1069μs (49%) — QK^T matmul dominates (8 heads × 64×1500)
- Self-attention: 823μs (38%) — K/V append + QK^T + attn@V
- MLP: 259μs (12%) — two Linear (512→2048, 2048→512) + GELU
- LayerNorm: 11μs (0.5%) — negligible

The 66 BLAS calls per token step contribute ~330μs of overhead (15% of total).
The remaining 85% is actual AMX compute — cannot be optimized in pure Java.

## 🎯 Remaining Optimization Opportunities

### Fused Decoder Block (would close the gap)

Write a single C function that executes an entire decoder block (self-attn + cross-attn + MLP) and call it via Panama
FFM. This eliminates 11 BLAS calls per block → 1 call per block. Estimated savings: ~250μs per block × 6 = 1.5ms per
token step.

### CoreML/ANE Offload (would exceed whisper.cpp)

The Apple Neural Engine can run the full decoder in ~1ms. Architecture: Java → Unix socket → Swift service → CoreML.
This would make the decoder faster than whisper.cpp's CPU path.

### Speculative Decoding

Generate multiple candidate tokens in parallel, verify in one forward pass. Amortizes BLAS overhead across N candidates.
Requires architectural changes to the decode loop.

### Quantized Inference (INT8/INT4)

Keep weights in quantized format and use quantized matmul. Reduces memory bandwidth (the real bottleneck for large
models). whisper.cpp supports Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 — our GGML loader already dequantizes these but we could keep
them quantized.

---

## Session: 2026-04-14 — Decoder Performance for Medium/Large Models

### Scoped Arena for Encoder (success — 68x → 1x overhead eliminated)

- Attempted: Per-layer `Arena.ofShared()` in WhisperEncoder.forward() to batch-free intermediate tensors
- Result: Encoder went from ~41s to ~3s per chunk for large-v3-turbo (32 layers, 1280 state)
- Why: `Arena.ofAuto()` relies on GC cleaners to free native memory. 32 encoder layers allocate ~10GB of intermediates
  per pass. The cleaner couldn't keep up, causing OS-level virtual memory thrashing. `Arena.ofShared()` frees everything
  instantly on `.close()`.
- Lesson: For tight loops allocating large native buffers, always use explicit arena lifecycle. `Arena.ofAuto()` is only
  suitable for long-lived allocations. Note: `Arena.ofConfined()` deadlocks when AccelerateBlas.softmaxRows spawns
  worker threads — must use `ofShared()`.

### M=1 SIMD Matmul Priority (success — decoder dispatch overhead eliminated)

- Attempted: Reordered matmul/matmulScaled/matmulTransB to check M==1 before AccelerateBlas.isAvailable()
- Result: Decoder per-token matmul calls use inline SIMD instead of BLAS sgemm
- Why: For M=1 (single token decode), BLAS sgemm has ~5μs FFM downcall overhead per call. With ~53 matmul calls per
  token (turbo), that's ~265μs wasted. The SIMD dot-product path has zero dispatch cost.
- Lesson: BLAS is only faster than SIMD for M>1 (actual matrix-matrix multiply). For matrix-vector (M=1), inline SIMD
  always wins.

### KV Cache Pre-allocated Buffer (success — medium.en 233s → 46s)

- Attempted: 2x-growth buffer for KV cache append, avoiding Arena.ofAuto() allocation per step
- Result: Decoder time for medium.en dropped from 198s to 26s (7.6x speedup)
- Why: Original appendHeadsCache allocated a new tensor via Arena.ofAuto() on every decode step, copying all existing
  data plus new data. The Arena allocation overhead dominated — not the O(n²) copy itself, but the native memory
  allocation/deallocation churn. The buffer approach reuses the backing store and only reallocates on 2x growth.
- Lesson: Arena.ofAuto() allocation is expensive in tight loops. Pre-allocate with growth factor for append-heavy
  patterns.

### Fused SIMD Attention Kernel (marginal — ~5% decoder improvement)

- Attempted: Combined QK^T + softmax + AV into one SIMD-vectorized pass per head for qLen==1
- Result: medium.en decoder 26s → 23s. Eliminates 5 tensor allocations per attention call.
- Why: The attention compute (headDim=64 dot products) is small relative to the Linear projection compute (state=1024
  matmuls). The allocation savings help but the compute savings are modest.
- Lesson: Fused kernels help most when the fused operations dominate total compute. For Whisper, Linear projections (
  MLP, Q/K/V/out) dominate decoder time, not attention.

### Smart Temperature Fallback (success — base.en 114s → 32s, turbo 626s → 136s)

- Attempted: Early exit from temperature fallback when t=0.0 has OK compression ratio, plus best-result tracking across
  temperatures
- Result: base.en 3.6x speedup, turbo 4.6x speedup. All models above real-time.
- Why: Greedy decoding produces avgLogprob marginally below -1.0 threshold, triggering all 6 fallback retries. Higher
  temperatures produce worse results (-9.9 avgLogprob). The original code used the last (worst) result. Now it keeps the
  best and exits early when higher temps degrade.
- Lesson: Temperature fallback is the single largest performance variable for smaller models. A bad fallback policy can
  6x the decode time with zero quality benefit.

### Decoder logits.getRow() (success — eliminated full tensor copy)

- Attempted: Read only the last vocab row from logits tensor instead of copying entire (1, seqLen, 51866) tensor
- Result: Eliminated ~200KB allocation per decode step
- Why: `logits.data()` copied the entire tensor from off-heap to heap. Only the last row (51866 floats) was needed.
- Lesson: Always check if you're copying more data than needed from off-heap tensors.

### LayerNorm Cached Gamma/Beta (success — eliminated 96 copies per encoder pass)

- Attempted: Pre-cache gamma.data() and beta.data() float arrays at LayerNorm construction time
- Result: Eliminated 96 off-heap-to-heap copies per encoder pass (3 layerNorms × 32 layers)
- Why: gamma/beta weights never change but were being copied from off-heap on every forward() call
- Lesson: Cache immutable weight data at construction time, not on every forward pass.

### Linear addInPlace Bias (success — eliminated 192 allocations per encoder pass)

- Attempted: Changed `out.add(bias)` to `out.addInPlace(bias)` in Linear.forward()
- Result: Eliminated one tensor allocation per Linear call
- Why: The matmul output tensor is freshly allocated and owned — safe to modify in-place
- Lesson: Use in-place operations when the input tensor won't be reused.

### Final Performance Summary (physicsworks.wav, 203s audio, Apple Silicon)

| Model                 | Before | After | Speedup | RTF   |
|-----------------------|--------|-------|---------|-------|
| tiny.en (39M)         | ~36s   | 13s   | 2.8x    | 15.6x |
| base.en (74M)         | ~126s  | 32s   | 3.9x    | 6.3x  |
| small.en (244M)       | ~45s   | 15s   | 3.0x    | 13.5x |
| medium.en (769M)      | 233s   | 43s   | 5.4x    | 4.7x  |
| large-v3-turbo (1.5G) | 626s   | 136s  | 4.6x    | 1.5x  |

### KV Cache Direct Copy (success — fixed 8.9 drift to 0.02)

- Attempted: Reverted pre-allocated gapped buffer back to direct copy (torch.cat equivalent)
- Result: Drift dropped from 8.9/step to 0.02/step. All tokens match Python reference.
- Why: The gapped buffer had stride mismatch — heads were stored at bufCap stride but matmul expected totalLen stride.
  The compact copy was reading from wrong offsets.
- Lesson: Any KV cache buffer optimization MUST pass the drift regression test (threshold 0.03). The Python reference
  uses torch.cat (allocates new tensor each step) — match that exactly before optimizing.

### BLAS sgemm for M=1 Matmul (success — 29% decoder speedup)

- Attempted: Route M=1 matmul/matmulScaled/matmulTransB through cblas_sgemm instead of pure SIMD dot products
- Result: All models faster. Turbo: 124.7s → 88.5s (1.6x → 2.3x RTF). Drift improved to 0.0187.
- Why: On Apple Silicon, cblas_sgemm dispatches to the AMX coprocessor even for M=1. AMX handles large K×N (1280×1280
  for turbo) much faster than SIMD vector loops. The original "BLAS dispatch overhead" concern was wrong for Apple
  Silicon — AMX throughput dominates the ~5μs dispatch cost.
- Lesson: Always benchmark BLAS vs SIMD for M=1. AMX coprocessors change the calculus — dispatch overhead is negligible
  compared to AMX throughput for K≥256.

### Updated Performance Summary (physicsworks.wav, 203s audio, greedy beam=1)

| Model                 | Time   | RTF   | vs Previous |
|-----------------------|--------|-------|-------------|
| tiny.en (39M)         | 8.6s   | 23.6x | 1.5x faster |
| base.en (74M)         | 12.4s  | 16.5x | 1.4x faster |
| small.en (244M)       | 28.3s  | 7.2x  | 1.3x faster |
| medium.en (769M)      | 103.5s | 2.0x  | 1.1x faster |
| large-v3-turbo (1.5G) | 88.5s  | 2.3x  | 1.4x faster |

### cblas_sgemv for M=1 (no-op — no measurable difference)

- Attempted: Microbenchmarked cblas_sgemv vs cblas_sgemm(M=1) across all model dimensions (384→51866)
- Result: Within 2% at turbo dimensions (1280×1280, 1280×5120). No consistent winner.
- Why: Apple Accelerate likely detects M=1 inside sgemm and dispatches to the same AMX microcode as sgemv.
- Lesson: Don't add a separate sgemv binding. sgemm(M=1) is already optimal on Apple Silicon. No further action needed.

### In-place GELU + Residual Add + Virtual Threads (success — 14% overall speedup)

- Attempted: Three encoder pipeline optimizations:
    1. `geluInPlace()` — MLP output is fresh from matmul, safe to GELU in-place (saves 960MB for turbo)
    2. `attnOut[0].addInPlace(x)` — attention output is fresh, safe for residual add (saves 234MB for turbo)
    3. `Thread.ofVirtual()` for softmax/GELU parallelization (eliminates 320 platform thread creates per encoder pass)
- Result: turbo 88.5s → 76.4s (2.3x → 2.7x RTF). All models improved 6-17%.
- Why: Reduced allocation pressure lets the scoped arena free less memory per layer. Virtual threads eliminate OS thread
  creation overhead.
- Lesson: AccelerateBlas.geluInPlace (vDSP sigmoid approximation) does NOT match the lookup table GELU — caused 10.1
  drift. Always use the lookup table for GELU.

---

## Next Steps — Decoder Optimization Opportunities

Per-step decoder profile for turbo (qLen=1, nState=1280, nHead=20, headDim=64, 4 blocks):

| Operation                                           | Per step       | Uses BLAS/SIMD | Notes                                           |
|-----------------------------------------------------|----------------|----------------|-------------------------------------------------|
| Linear projections (Q/K/V/Out × 4 blocks + MLP × 4) | 24 sgemm calls | ✅ AMX          | Largest: mlp1 (1×5120×1280)                     |
| Cross-attention fused kernel (4 blocks × 1500 KV)   | 4 calls        | ✅ SIMD         | Dominates attention time                        |
| Self-attention fused kernel (4 blocks × growing KV) | 4 calls        | ✅ SIMD         | Grows with sequence                             |
| Logits projection (1×1280 @ 51866×1280^T)           | 1 sgemm        | ✅ AMX          | 133M FLOPs — single most expensive op           |
| LayerNorm (13 per step)                             | 13 calls       | ❌ scalar       | Allocates new tensor each call                  |
| GELU (4 per step)                                   | 4 calls        | ✅ F16 LUT      | In-place, no allocation                         |
| KV cache append (8 entries)                         | 8 memcpy       | ❌ O(n²) total  | torch.cat equivalent — copies all existing data |

### Ranked by expected impact

1. **Logits projection shortcut** (high impact, medium risk)
   Every step computes full (1,51866) logits just to pick one token. Speculative approaches:
   project only against top-K candidates from previous step, or use a smaller projection head
   for early exit. Would cut ~30% of per-step time.

2. **KV cache pre-allocated buffer** (medium impact, high risk)
   Current O(n²) copy matches Python's torch.cat. A pre-allocated buffer with 2x growth would
   make append O(1) amortized. Previous attempts caused 8.9 drift due to stride mismatch.
   Any implementation MUST pass the drift regression test (threshold 0.03).

3. **LayerNorm in-place + SIMD** (low-medium impact, low risk)
   Currently scalar loops + allocates new tensor per call (13 per step). Input tensor is always
   fresh from a previous op — safe to mutate in-place. Vectorize mean/variance/normalize with
   SIMD or vDSP (vDSP_meanv, vDSP_measqv). Saves 13 allocations per step.

4. **Batched decoder Linear ops** (medium impact, medium effort)
   All 4 decoder blocks run sequentially. Could batch Q/K/V projections across blocks into a
   single sgemm call with larger M dimension. Reduces BLAS dispatch overhead from 24 calls to ~6.

5. **MLX / Apple ML Compute backend** (high impact, very high effort)
   Apple's MLX framework provides GPU-accelerated matmul on Apple Silicon. Would require a
   new backend alongside AccelerateBlas. Potential 5-10x speedup for large models but adds
   significant complexity and platform dependency.

6. **Fused softmax in attention kernel** (low impact, low risk)
   The fused attention kernel uses scalar Math.exp for softmax. Could use vDSP_vvexpf for
   the exp step. Marginal gain since kvLen is small for self-attention (grows with sequence).

---

## Session: 2026-04-15 — Multi-Release JAR (Java 25 base, Java 26 Vector API overlay)

### Architecture Change: Vector API Made Optional

Refactored whisper4j into a multi-release JAR so it runs on Java 25+ without `--enable-preview`:

- **Base (Java 25):** All code compiles without Vector API. `SimdOps` provides scalar fallback loops.
- **Overlay (Java 26):** `VectorSimdOps` in `META-INF/versions/26/` uses `jdk.incubator.vector` for SIMD.
- **Detection:** `SimdOps` static initializer tries `Class.forName("jdk.incubator.vector.FloatVector")` then
  reflectively instantiates `VectorSimdOps`. Falls back to scalar on failure.

### Source Layout

```
src/main/java/     → base (Java 25, no --enable-preview)
src/main/java26/   → overlay (Java 26, --enable-preview --add-modules jdk.incubator.vector)
```

### Build

```bash
./mvnw compile          # base Java 25 + overlay Java 26 → multi-release JAR
./mvnw test             # tests run WITHOUT --enable-preview (scalar path)
./mvnw test -Pvector-api  # tests run WITH --enable-preview (SIMD path)
```

### Performance: Java 25 (scalar) vs Java 26 (Vector API SIMD) — base.en model

Test: `ValidateAgainstReferenceTest` — 203s audio (physicsworks.wav), greedy beam=1, Apple Silicon M2 Max.

| Path             | Total Time | Encoder (avg) | Decoder (avg) | Notes                            |
|------------------|------------|---------------|---------------|----------------------------------|
| Java 25 (scalar) | ~163s      | ~310ms        | ~2800ms       | No Vector API, BLAS still active |
| Java 26 (SIMD)   | ~171s      | ~310ms        | ~3200ms       | Vector API enabled               |

**Result: No meaningful difference on Apple Silicon with Accelerate BLAS.**

### Why Vector API Doesn't Help Here

On Apple Silicon, the performance hierarchy is:

1. **Apple Accelerate BLAS (AMX coprocessor)** — handles all matmul ops (~85% of compute)
2. **F16 GELU lookup table** — handles activation functions (no SIMD needed)
3. **Scalar loops** — handles element-wise add, scale, layerNorm, softmax

The Vector API accelerates category 3, which is <5% of total compute. The SIMD overhead
(JIT warmup, vector lane management, incubator module loading) slightly exceeds the savings
for these small operations.

### When Vector API WILL Help

- **Linux/Windows without BLAS:** matmul falls back to tiled scalar loops. The SIMD dot product
  and matmulVec paths provide 2-4x speedup for these operations.
- **Non-Apple platforms:** No AMX coprocessor, so SIMD is the fastest available path for
  element-wise ops.
- **Future:** When Vector API graduates from incubator (no `--enable-preview` needed), the
  JIT overhead will decrease and the SIMD path may become net-positive on all platforms.

### Lesson

The multi-release JAR is the right architecture regardless of current performance numbers.
It ensures whisper4j runs on Java 25+ (broadest compatibility) while automatically using
SIMD when available. The performance difference is negligible on Apple Silicon but will
matter on other platforms.

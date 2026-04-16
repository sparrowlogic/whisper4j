package com.sparrowlogic.whisper4j.model;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import com.sparrowlogic.whisper4j.annotation.Nullable;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Loads Whisper model weights from whisper.cpp GGML binary format.
 * Uses Panama Foreign Memory API — model weights live off-heap via mmap.
 * F32 tensors are zero-copy slices of the mmap'd file.
 * Quantized tensors are dequantized into arena-allocated off-heap memory.
 */
public final class GgmlLoader implements ModelLoader {

    private static final Logger LOG = Logger.getLogger(GgmlLoader.class.getName());
    private static final int GGML_MAGIC = 0x67676D6C;

    private static final int GGML_TYPE_F32  = 0;
    private static final int GGML_TYPE_F16  = 1;
    private static final int GGML_TYPE_Q4_0 = 2;
    private static final int GGML_TYPE_Q4_1 = 3;
    private static final int GGML_TYPE_Q5_0 = 6;
    private static final int GGML_TYPE_Q5_1 = 7;
    private static final int GGML_TYPE_Q8_0 = 8;

    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private @Nullable Map<Integer, String> vocab;
    private float @Nullable [] melFilters;
    private int melFilterNMel;
    private int melFilterNFft;
    private @Nullable ModelDimensions dims;
    private @Nullable Arena arena;

    @Override
    public WeightStore load(Path path) throws IOException {
        long fileSize = path.toFile().length();
        LOG.info("Loading GGML model from " + path + " (" + (fileSize / 1_000_000) + " MB)");
        logMemory("before load");

        arena = Arena.ofShared();
        try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
            MemorySegment mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
            WeightStore store = parse(mapped, fileSize);
            logMemory("after load");
            return store;
        } catch (OutOfMemoryError | IOException | IllegalArgumentException e) {
            LOG.severe("Failed to load model: " + e.getMessage());
            close();
            throw new IOException("Model load failed — insufficient memory or corrupt file: " + path, e);
        } catch (Exception e) {
            LOG.severe("Unexpected error loading model: " + e.getMessage());
            close();
            throw new IOException("Model load failed: " + path, e);
        }
    }

    /** Transfer arena ownership to the caller (e.g., WhisperModel). Returns null if already closed. */
    public @Nullable Arena takeArena() {
        Arena a = this.arena;
        this.arena = null;
        return a;
    }

    public void close() {
        if (arena != null) {
            try {
                arena.close();
            } catch (IllegalStateException e) {
                LOG.fine("Arena already closed: " + e.getMessage());
            }
            arena = null;
        }
    }
    public @Nullable Map<Integer, String> vocab() { return this.vocab; }
    public float @Nullable [] melFilters() { return this.melFilters; }
    public int melFilterNMel() { return this.melFilterNMel; }
    public int melFilterNFft() { return this.melFilterNFft; }
    public @Nullable ModelDimensions dimensions() { return this.dims; }

    private WeightStore parse(MemorySegment seg, long fileSize) {
        long pos = 0;

        int magic = seg.get(INT_LE, pos); pos += 4;
        if (magic != GGML_MAGIC) throw new IllegalArgumentException("not a GGML file (magic: 0x%08X)".formatted(magic));

        // hparams
        int nVocab      = seg.get(INT_LE, pos); pos += 4;
        int nAudioCtx   = seg.get(INT_LE, pos); pos += 4;
        int nAudioState = seg.get(INT_LE, pos); pos += 4;
        int nAudioHead  = seg.get(INT_LE, pos); pos += 4;
        int nAudioLayer = seg.get(INT_LE, pos); pos += 4;
        int nTextCtx    = seg.get(INT_LE, pos); pos += 4;
        int nTextState  = seg.get(INT_LE, pos); pos += 4;
        int nTextHead   = seg.get(INT_LE, pos); pos += 4;
        int nTextLayer  = seg.get(INT_LE, pos); pos += 4;
        int nMels       = seg.get(INT_LE, pos); pos += 4;
        int ftype       = seg.get(INT_LE, pos); pos += 4;

        dims = new ModelDimensions(nMels, nAudioCtx, nAudioState, nAudioHead, nAudioLayer,
                nVocab, nTextCtx, nTextState, nTextHead, nTextLayer);
        LOG.info("GGML hparams: n_vocab=%d n_audio_ctx=%d n_audio_state=%d n_audio_head=%d n_audio_layer=%d n_text_ctx=%d n_text_state=%d n_text_head=%d n_text_layer=%d n_mels=%d ftype=%d"
                .formatted(nVocab, nAudioCtx, nAudioState, nAudioHead, nAudioLayer, nTextCtx, nTextState, nTextHead, nTextLayer, nMels, ftype));

        // mel filters
        melFilterNMel = seg.get(INT_LE, pos); pos += 4;
        melFilterNFft = seg.get(INT_LE, pos); pos += 4;
        melFilters = new float[melFilterNMel * melFilterNFft];
        MemorySegment.copy(seg, FLOAT_LE, pos, melFilters, 0, melFilters.length);
        pos += (long) melFilters.length * Float.BYTES;

        // vocab
        int vocabCount = seg.get(INT_LE, pos); pos += 4;
        vocab = new LinkedHashMap<>();
        for (int i = 0; i < vocabCount; i++) {
            int len = seg.get(INT_LE, pos); pos += 4;
            if (len > 0) {
                byte[] bytes = new byte[len];
                MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, pos, bytes, 0, len);
                pos += len;
                vocab.put(i, new String(bytes, StandardCharsets.UTF_8));
            } else {
                vocab.put(i, "");
            }
        }
        LOG.info("Loaded vocab: %d tokens".formatted(vocab.size()));
        LOG.info("Tensor data starts at file position: %d (%.1f MB into file)".formatted(pos, pos / 1e6));

        // tensors
        WeightStore store = new WeightStore();
        long totalBytes = 0;
        int tensorCount = 0;

        while (pos + 12 <= fileSize) {
            int nDims   = seg.get(INT_LE, pos); pos += 4;
            int nameLen = seg.get(INT_LE, pos); pos += 4;
            int ttype   = seg.get(INT_LE, pos); pos += 4;

            // sanity check — GGML tensors have 1-4 dims, names < 256 chars, types 0-8
            if (nDims < 1 || nDims > 4 || nameLen < 1 || nameLen > 256 || ttype < 0 || ttype > 15) {
                LOG.warning("Tensor header looks invalid at pos=%d (nDims=%d nameLen=%d ttype=%d) — likely end of tensors or alignment issue"
                        .formatted(pos - 12, nDims, nameLen, ttype));
                break;
            }

            int[] ne = new int[nDims];
            int nElements = 1;
            for (int i = 0; i < nDims; i++) {
                ne[i] = seg.get(INT_LE, pos); pos += 4;
                nElements *= ne[i];
            }

            byte[] nameBytes = new byte[nameLen];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, pos, nameBytes, 0, nameLen);
            pos += nameLen;
            String name = new String(nameBytes, StandardCharsets.UTF_8).trim();

            if (tensorCount == 0) {
                LOG.info("First tensor: '%s' nDims=%d ttype=%d ne=%s at filePos=%d"
                        .formatted(name, nDims, ttype, java.util.Arrays.toString(ne), pos));
            }

            // NOTE: old GGML format has NO alignment padding — data follows name immediately
            long dataSize = tensorDataSize(ttype, nElements);

            // GGML stores column-major. Column-major (d0,d1,...,dn) has identical
            // memory layout to row-major (dn,...,d1,d0). Just reverse the shape.
            int[] shape = new int[nDims];
            for (int i = 0; i < nDims; i++) shape[i] = ne[nDims - 1 - i];

            Tensor tensor;
            if (ttype == GGML_TYPE_F32) {
                // zero-copy slice of mmap'd file
                tensor = Tensor.ofSegment(seg.asSlice(pos, dataSize), shape);
            } else {
                // dequantize into off-heap arena memory
                long allocBytes = (long) nElements * Float.BYTES;
                MemorySegment outSeg;
                try {
                    outSeg = arena.allocate(allocBytes, Float.BYTES);
                } catch (OutOfMemoryError oom) {
                    LOG.severe("Out of memory dequantizing tensor '%s' (%d elements, need %d MB). Loaded %d tensors before failure."
                            .formatted(name, nElements, allocBytes / (1024 * 1024), tensorCount));
                    throw oom; // caught by load()
                }
                dequantize(seg, pos, ttype, nElements, outSeg);
                tensor = Tensor.ofSegment(outSeg, shape);
            }
            pos += dataSize;

            store.put(name, tensor);
            totalBytes += (long) nElements * Float.BYTES;
            tensorCount++;
            if (tensorCount % 50 == 0) {
                long pct = pos * 100 / fileSize;
                Runtime rt = Runtime.getRuntime();
                long heapMB = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
                LOG.info("  loading tensors... %d (%d%% of file, heap %d MB)"
                        .formatted(tensorCount, pct, heapMB));
            }
            LOG.fine("  tensor %-60s type=%d shape=%s".formatted(name, ttype, java.util.Arrays.toString(shape)));
        }

        LOG.info("Loaded %d tensors (%.1f MB as f32, off-heap) — model ready".formatted(tensorCount, totalBytes / 1e6));
        return store;
    }

    // ---- Dequantization (reads from mmap segment, writes to arena segment) ----

    private static void dequantize(MemorySegment src, long srcPos, int ttype, int nElements, MemorySegment dst) {
        switch (ttype) {
            case GGML_TYPE_F16 -> {
                for (int i = 0; i < nElements; i++) {
                    dst.setAtIndex(FLOAT_LE, i, Float.float16ToFloat(src.get(SHORT_LE, srcPos + (long) i * 2)));
                }
            }
            case GGML_TYPE_Q4_0 -> dequantizeQ4_0(src, srcPos, dst, nElements);
            case GGML_TYPE_Q4_1 -> dequantizeQ4_1(src, srcPos, dst, nElements);
            case GGML_TYPE_Q5_0 -> dequantizeQ5_0(src, srcPos, dst, nElements);
            case GGML_TYPE_Q5_1 -> dequantizeQ5_1(src, srcPos, dst, nElements);
            case GGML_TYPE_Q8_0 -> dequantizeQ8_0(src, srcPos, dst, nElements);
            default -> throw new IllegalArgumentException("unsupported GGML type: " + ttype);
        }
    }

    private static void dequantizeQ4_0(MemorySegment src, long off, MemorySegment dst, int nElements) {
        int nBlocks = nElements / 32;
        for (int b = 0; b < nBlocks; b++) {
            float scale = Float.float16ToFloat(src.get(SHORT_LE, off)); off += 2;
            for (int j = 0; j < 16; j++) {
                int v = src.get(ValueLayout.JAVA_BYTE, off++) & 0xFF;
                dst.setAtIndex(FLOAT_LE, b * 32 + j,      ((v & 0x0F) - 8) * scale);
                dst.setAtIndex(FLOAT_LE, b * 32 + j + 16, (((v >> 4) & 0x0F) - 8) * scale);
            }
        }
    }

    private static void dequantizeQ4_1(MemorySegment src, long off, MemorySegment dst, int nElements) {
        int nBlocks = nElements / 32;
        for (int b = 0; b < nBlocks; b++) {
            float scale = Float.float16ToFloat(src.get(SHORT_LE, off)); off += 2;
            float min = Float.float16ToFloat(src.get(SHORT_LE, off)); off += 2;
            for (int j = 0; j < 16; j++) {
                int v = src.get(ValueLayout.JAVA_BYTE, off++) & 0xFF;
                dst.setAtIndex(FLOAT_LE, b * 32 + j,      (v & 0x0F) * scale + min);
                dst.setAtIndex(FLOAT_LE, b * 32 + j + 16, ((v >> 4) & 0x0F) * scale + min);
            }
        }
    }

    private static void dequantizeQ5_0(MemorySegment src, long off, MemorySegment dst, int nElements) {
        int nBlocks = nElements / 32;
        for (int b = 0; b < nBlocks; b++) {
            float scale = Float.float16ToFloat(src.get(SHORT_LE, off)); off += 2;
            int qh = src.get(INT_LE, off); off += 4;
            for (int j = 0; j < 16; j++) {
                int v = src.get(ValueLayout.JAVA_BYTE, off++) & 0xFF;
                dst.setAtIndex(FLOAT_LE, b * 32 + j,      (((v & 0x0F) | (((qh >> j) & 1) << 4)) - 16) * scale);
                dst.setAtIndex(FLOAT_LE, b * 32 + j + 16, ((((v >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4)) - 16) * scale);
            }
        }
    }

    private static void dequantizeQ5_1(MemorySegment src, long off, MemorySegment dst, int nElements) {
        int nBlocks = nElements / 32;
        for (int b = 0; b < nBlocks; b++) {
            float scale = Float.float16ToFloat(src.get(SHORT_LE, off)); off += 2;
            float min = Float.float16ToFloat(src.get(SHORT_LE, off)); off += 2;
            int qh = src.get(INT_LE, off); off += 4;
            for (int j = 0; j < 16; j++) {
                int v = src.get(ValueLayout.JAVA_BYTE, off++) & 0xFF;
                dst.setAtIndex(FLOAT_LE, b * 32 + j,      ((v & 0x0F) | (((qh >> j) & 1) << 4)) * scale + min);
                dst.setAtIndex(FLOAT_LE, b * 32 + j + 16, (((v >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4)) * scale + min);
            }
        }
    }

    private static void dequantizeQ8_0(MemorySegment src, long off, MemorySegment dst, int nElements) {
        int nBlocks = nElements / 32;
        for (int b = 0; b < nBlocks; b++) {
            float scale = Float.float16ToFloat(src.get(SHORT_LE, off)); off += 2;
            for (int j = 0; j < 32; j++) {
                dst.setAtIndex(FLOAT_LE, b * 32 + j, src.get(ValueLayout.JAVA_BYTE, off++) * scale);
            }
        }
    }

    private static long tensorDataSize(int ttype, int nElements) {
        return switch (ttype) {
            case GGML_TYPE_F32  -> (long) nElements * 4;
            case GGML_TYPE_F16  -> (long) nElements * 2;
            case GGML_TYPE_Q4_0 -> (long) (nElements / 32) * 18;
            case GGML_TYPE_Q4_1 -> (long) (nElements / 32) * 20;
            case GGML_TYPE_Q5_0 -> (long) (nElements / 32) * 22;
            case GGML_TYPE_Q5_1 -> (long) (nElements / 32) * 24;
            case GGML_TYPE_Q8_0 -> (long) (nElements / 32) * 34;
            default -> throw new IllegalArgumentException("unsupported GGML type: " + ttype);
        };
    }

    private static void logMemory(String label) {
        Runtime rt = Runtime.getRuntime();
        long used = rt.totalMemory() - rt.freeMemory();
        LOG.info("  memory [%s]: heap used=%d MB, committed=%d MB, max=%d MB"
                .formatted(label, used / (1024 * 1024), rt.totalMemory() / (1024 * 1024), rt.maxMemory() / (1024 * 1024)));
    }
}

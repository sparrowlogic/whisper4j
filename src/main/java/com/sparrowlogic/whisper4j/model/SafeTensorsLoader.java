package com.sparrowlogic.whisper4j.model;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import org.jspecify.annotations.Nullable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Loads weights from HuggingFace SafeTensors format.
 * Format: 8-byte LE uint64 header length, JSON header, then raw binary tensor data.
 */
public final class SafeTensorsLoader implements ModelLoader {

    private static final Logger LOG = Logger.getLogger(SafeTensorsLoader.class.getName());

    @Override
    public WeightStore load(final Path path) throws IOException {
        LOG.info("Loading SafeTensors from " + path);
        try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            return parse(buf);
        }
    }

    static WeightStore parse(final ByteBuffer buf) {
        long headerLen = buf.getLong();
        byte[] headerBytes = new byte[(int) headerLen];
        buf.get(headerBytes);
        int dataOffset = 8 + (int) headerLen;
        String header = new String(headerBytes, StandardCharsets.UTF_8).trim();

        WeightStore store = new WeightStore();
        Map<String, TensorMeta> metas = parseHeader(header);

        for (var entry : metas.entrySet()) {
            String name = entry.getKey();
            TensorMeta meta = entry.getValue();
            if ("__metadata__".equals(name)) {
                continue;
            }

            int start = dataOffset + meta.offsetStart;
            int numBytes = meta.offsetEnd - meta.offsetStart;
            buf.position(start);

            float[] data = switch (meta.dtype) {
                case "F32" -> readF32(buf, numBytes);
                case "F16" -> readF16(buf, numBytes);
                case "BF16" -> readBF16(buf, numBytes);
                default -> throw new IllegalArgumentException("unsupported dtype: " + meta.dtype);
            };
            store.put(name, Tensor.of(data, meta.shape));
            LOG.fine("  tensor %-60s dtype=%s shape=%s"
                    .formatted(name, meta.dtype, Arrays.toString(meta.shape)));
        }
        LOG.info("Loaded %d tensors from SafeTensors".formatted(store.size()));
        return store;
    }

    private static float[] readF32(final ByteBuffer buf, final int numBytes) {
        float[] f = new float[numBytes / 4];
        buf.asFloatBuffer().get(f);
        return f;
    }

    private static float[] readF16(final ByteBuffer buf, final int numBytes) {
        float[] f = new float[numBytes / 2];
        for (int i = 0; i < f.length; i++) {
            f[i] = Float.float16ToFloat(buf.getShort());
        }
        return f;
    }

    private static float[] readBF16(final ByteBuffer buf, final int numBytes) {
        float[] f = new float[numBytes / 2];
        for (int i = 0; i < f.length; i++) {
            f[i] = Float.intBitsToFloat((buf.getShort() & 0xFFFF) << 16);
        }
        return f;
    }

    // ---- Minimal JSON parsing for SafeTensors header ----

    private record TensorMeta(String dtype, int[] shape, int offsetStart, int offsetEnd) { }

    private static Map<String, TensorMeta> parseHeader(final String json) {
        Map<String, TensorMeta> result = new LinkedHashMap<>();
        int pos = json.indexOf('{') + 1;
        while (pos < json.length()) {
            int keyStart = json.indexOf('"', pos);
            if (keyStart < 0) {
                break;
            }
            int keyEnd = findStringEnd(json, keyStart);
            String key = json.substring(keyStart + 1, keyEnd);
            int valStart = json.indexOf('{', keyEnd);
            if (valStart < 0) {
                break;
            }
            int valEnd = findMatchingBrace(json, valStart);
            if (!"__metadata__".equals(key)) {
                String val = json.substring(valStart, valEnd + 1);
                String dtype = extractString(val, "dtype");
                int[] shape = extractIntArray(val, "shape");
                int[] offsets = extractIntArray(val, "data_offsets");
                result.put(key, new TensorMeta(dtype, shape, offsets[0], offsets[1]));
            }
            pos = valEnd + 1;
        }
        return result;
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
        boolean inStr = false;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\' && inStr) {
                i++;
                continue;
            }
            if (c == '"') {
                inStr = !inStr;
            }
            if (!inStr) {
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

    private static @Nullable String extractString(final String json, final String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return null;
        }
        int colon = json.indexOf(':', idx);
        int strStart = json.indexOf('"', colon + 1);
        int strEnd = findStringEnd(json, strStart);
        return json.substring(strStart + 1, strEnd);
    }

    private static int[] extractIntArray(final String json, final String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return new int[0];
        }
        int arrStart = json.indexOf('[', idx);
        int arrEnd = json.indexOf(']', arrStart);
        String inner = json.substring(arrStart + 1, arrEnd).trim();
        if (inner.isEmpty()) {
            return new int[0];
        }
        return Arrays.stream(inner.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
    }
}

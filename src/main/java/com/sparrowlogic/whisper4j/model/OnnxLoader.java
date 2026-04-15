package com.sparrowlogic.whisper4j.model;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Loads weights from ONNX model files.
 * ONNX uses protobuf encoding. This implements a minimal protobuf wire format reader
 * sufficient to extract initializer tensors — no protobuf dependency needed.
 *
 * <p>ONNX protobuf structure (simplified):
 *   ModelProto { graph: GraphProto { initializer: [TensorProto, ...] } }
 *   TensorProto { dims, data_type, name, raw_data, float_data, ... }
 */
@SuppressWarnings({"checkstyle:ExecutableStatementCount", "checkstyle:MissingSwitchDefault"})
public final class OnnxLoader implements ModelLoader {

    private static final Logger LOG = Logger.getLogger(OnnxLoader.class.getName());

    // ONNX TensorProto data types
    private static final int ONNX_FLOAT = 1;
    private static final int ONNX_DOUBLE = 11;
    private static final int ONNX_FLOAT16 = 10;
    private static final int ONNX_BFLOAT16 = 16;
    private static final int ONNX_INT32 = 6;
    private static final int ONNX_INT64 = 7;

    @Override
    public WeightStore load(final Path path) throws IOException {
        LOG.info("Loading ONNX model from " + path);

        // Check for split encoder/decoder ONNX (HuggingFace ONNX community format)
        Path dir = Files.isDirectory(path) ? path : path.getParent();
        Path onnxDir = dir != null && Files.exists(dir.resolve("encoder_model.onnx"))
                ? dir : (dir != null && Files.exists(dir.resolve("onnx/encoder_model.onnx"))
                ? dir.resolve("onnx") : null);

        if (onnxDir != null) {
            return loadSplit(onnxDir);
        }

        // Single ONNX file
        try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            return parseModel(buf);
        }
    }

    private WeightStore loadSplit(final Path onnxDir) throws IOException {
        WeightStore store = new WeightStore();

        // Load encoder
        Path encPath = onnxDir.resolve("encoder_model.onnx");
        LOG.info("Loading ONNX encoder from " + encPath);
        try (var ch = FileChannel.open(encPath, StandardOpenOption.READ)) {
            ByteBuffer buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            WeightStore enc = parseModel(buf);
            for (String name : enc.names()) {
                // ONNX names may already have model.encoder. prefix, or just encoder.
                // If no encoder/decoder prefix, add encoder.
                String key = name.startsWith("model.") || name.startsWith("encoder.") ? name : "encoder." + name;
                store.put(key, enc.get(name));
            }
        }

        // Load decoder
        Path decPath = onnxDir.resolve("decoder_model.onnx");
        LOG.info("Loading ONNX decoder from " + decPath);
        try (var ch = FileChannel.open(decPath, StandardOpenOption.READ)) {
            ByteBuffer buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            WeightStore dec = parseModel(buf);
            for (String name : dec.names()) {
                String key = name.startsWith("model.") || name.startsWith("decoder.") ? name : "decoder." + name;
                store.put(key, dec.get(name));
            }
        }

        LOG.info("Loaded %d total tensors from split ONNX".formatted(store.size()));
        return store;
    }

    private static WeightStore parseModel(final ByteBuffer buf) {
        WeightStore store = new WeightStore();
        // ModelProto: field 7 = GraphProto
        try {
            while (buf.hasRemaining()) {
                long tag = readVarint(buf);
                int fieldNum = (int) (tag >>> 3);
                int wireType = (int) (tag & 0x7);
                if (fieldNum == 7 && wireType == 2) {
                    int len = (int) readVarint(buf);
                    int end = buf.position() + len;
                    parseGraph(buf, end, store);
                    buf.position(end);
                } else {
                    skipField(buf, wireType);
                }
            }
        } catch (Exception e) {
            LOG.warning("ONNX parse stopped: " + e.getMessage() + " (loaded " + store.size() + " tensors)");
        }
        LOG.info("Loaded %d tensors from ONNX".formatted(store.size()));
        return store;
    }

    private static void parseGraph(final ByteBuffer buf, final int end, final WeightStore store) {
        java.util.Map<Integer, Integer> fieldCounts = new java.util.TreeMap<>();
        while (buf.position() < end) {
            long tag = readVarint(buf);
            int fieldNum = (int) (tag >>> 3);
            int wireType = (int) (tag & 0x7);
            fieldCounts.merge(fieldNum, 1, Integer::sum);
            if (fieldNum == 5 && wireType == 2) {
                int len = (int) readVarint(buf);
                int tensorEnd = buf.position() + len;
                parseTensor(buf, tensorEnd, store);
                buf.position(tensorEnd);
            } else {
                skipField(buf, wireType);
            }
        }
        LOG.info("parseGraph field counts: %s, tensors loaded: %d".formatted(fieldCounts, store.size()));
    }

    @SuppressWarnings("checkstyle:CyclomaticComplexity")
    private static void parseTensor(final ByteBuffer buf, final int end, final WeightStore store) {
        List<Long> dims = new ArrayList<>();
        int dataType = 0;
        String name = null;
        byte[] rawData = null;
        List<Float> floatData = new ArrayList<>();

        while (buf.position() < end) {
            long tag = readVarint(buf);
            int fieldNum = (int) (tag >>> 3);
            int wireType = (int) (tag & 0x7);

            switch (fieldNum) {
                case 1 -> {
                    if (wireType == 0) {
                        dims.add(readVarint(buf));
                    } else if (wireType == 2) {
                        int len = (int) readVarint(buf);
                        int dimEnd = buf.position() + len;
                        while (buf.position() < dimEnd) {
                            dims.add(readVarint(buf));
                        }
                    }
                }
                case 2 -> dataType = (int) readVarint(buf);
                case 8 -> {
                    int len = (int) readVarint(buf);
                    byte[] bytes = new byte[len];
                    buf.get(bytes);
                    name = new String(bytes);
                }
                case 9 -> { // raw_data (field 9 in TensorProto)
                    int len = (int) readVarint(buf);
                    rawData = new byte[len];
                    buf.get(rawData);
                }
                case 4 -> {
                    if (wireType == 2) {
                        int len = (int) readVarint(buf);
                        int fEnd = buf.position() + len;
                        while (buf.position() < fEnd) {
                            floatData.add(buf.getFloat());
                        }
                    } else if (wireType == 5) {
                        floatData.add(buf.getFloat());
                    }
                }
                default -> skipField(buf, wireType);
            }
        }

        if (name == null || (rawData == null && floatData.isEmpty())) {
            return;
        }

        int[] shape = dims.stream().mapToInt(Long::intValue).toArray();
        int size = 1;
        for (int d : shape) {
            size *= d;
        }

        float[] data;
        if (!floatData.isEmpty()) {
            data = new float[floatData.size()];
            for (int i = 0; i < data.length; i++) {
                data[i] = floatData.get(i);
            }
        } else {
            data = decodeRawData(rawData, dataType, size);
        }

        store.put(name, Tensor.of(data, shape));
    }

    private static float[] decodeRawData(final byte[] raw, final int dataType, final int size) {
        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        float[] data = new float[size];
        switch (dataType) {
            case ONNX_FLOAT -> buf.asFloatBuffer().get(data);
            case ONNX_FLOAT16 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = Float.float16ToFloat(buf.getShort());
                }
            }
            case ONNX_BFLOAT16 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = Float.intBitsToFloat((buf.getShort() & 0xFFFF) << 16);
                }
            }
            case ONNX_DOUBLE -> {
                for (int i = 0; i < size; i++) {
                    data[i] = (float) buf.getDouble();
                }
            }
            case ONNX_INT32 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = buf.getInt();
                }
            }
            case ONNX_INT64 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = buf.getLong();
                }
            }
            default -> throw new IllegalArgumentException("unsupported ONNX dtype: " + dataType);
        }
        return data;
    }

    // ---- Protobuf wire format helpers ----

    private static long readVarint(final ByteBuffer buf) {
        long result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
        }
        return result;
    }

    private static void skipField(final ByteBuffer buf, final int wireType) {
        switch (wireType) {
            case 0 -> readVarint(buf);
            case 1 -> buf.position(buf.position() + 8);
            case 2 -> {
                int len = (int) readVarint(buf);
                buf.position(buf.position() + len);
            }
            case 5 -> buf.position(buf.position() + 4);
            default -> { }
        }
    }
}

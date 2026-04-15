package com.sparrowlogic.whisper4j.model;

import com.sparrowlogic.whisper4j.tensor.Tensor;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.logging.Logger;

/**
 * Loads weights from CTranslate2 model.bin format.
 * Binary format (from model_spec.py _serialize):
 *   uint32 binary_version, string model_name, uint32 spec_revision, uint32 num_variables,
 *   per variable: string name, uint8 ndims, uint32[] shape, uint8 dtype_id, uint32 num_bytes, raw data,
 *   uint32 num_aliases, per alias: string alias_name, string variable_name
 */
@SuppressWarnings("checkstyle:MissingSwitchDefault")
public final class CTranslate2Loader implements ModelLoader {

    private static final Logger LOG = Logger.getLogger(CTranslate2Loader.class.getName());

    // dtype IDs matching CTranslate2's DataType enum
    private static final int DTYPE_FLOAT32 = 0;
    private static final int DTYPE_INT8 = 1;
    private static final int DTYPE_INT16 = 2;
    private static final int DTYPE_INT32 = 3;
    private static final int DTYPE_FLOAT16 = 4;
    private static final int DTYPE_BFLOAT16 = 5;

    @Override
    public WeightStore load(final Path path) throws IOException {
        Path modelBin = path.toFile().isDirectory() ? path.resolve("model.bin") : path;
        LOG.info("Loading CTranslate2 model from " + modelBin);
        try (var ch = FileChannel.open(modelBin, StandardOpenOption.READ)) {
            ByteBuffer buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            return parse(buf);
        }
    }

    static WeightStore parse(final ByteBuffer buf) {
        WeightStore store = new WeightStore();

        int version = buf.getInt();
        String modelName = readString(buf);
        int revision = buf.getInt();
        int numVariables = buf.getInt();
        LOG.info("CT2 model: name=%s version=%d revision=%d variables=%d"
                .formatted(modelName, version, revision, numVariables));

        for (int i = 0; i < numVariables; i++) {
            String name = readString(buf);
            int ndims = buf.get() & 0xFF;
            int[] shape = new int[ndims];
            int size = 1;
            for (int d = 0; d < ndims; d++) {
                shape[d] = buf.getInt();
                size *= shape[d];
            }
            int dtypeId = buf.get() & 0xFF;
            int numBytes = buf.getInt();

            float[] data = readTensorData(buf, dtypeId, numBytes, size);
            // convert CT2 name format (slash-separated) to dot-separated
            String javaName = name.replace('/', '.');
            store.put(javaName, Tensor.of(data, shape));
        }

        // read aliases
        int numAliases = buf.getInt();
        for (int i = 0; i < numAliases; i++) {
            String aliasName = readString(buf).replace('/', '.');
            String targetName = readString(buf).replace('/', '.');
            if (store.contains(targetName)) {
                store.put(aliasName, store.get(targetName));
            }
        }

        return store;
    }

    private static float[] readTensorData(final ByteBuffer buf, final int dtypeId,
                                          final int numBytes, final int size) {
        float[] data = new float[size];
        int startPos = buf.position();
        switch (dtypeId) {
            case DTYPE_FLOAT32 -> buf.asFloatBuffer().get(data, 0, size);
            case DTYPE_FLOAT16 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = Float.float16ToFloat(buf.getShort());
                }
                buf.position(startPos);
            }
            case DTYPE_BFLOAT16 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = Float.intBitsToFloat((buf.getShort() & 0xFFFF) << 16);
                }
                buf.position(startPos);
            }
            case DTYPE_INT8 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = buf.get();
                }
                buf.position(startPos);
            }
            case DTYPE_INT16 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = buf.getShort();
                }
                buf.position(startPos);
            }
            case DTYPE_INT32 -> {
                for (int i = 0; i < size; i++) {
                    data[i] = buf.getInt();
                }
                buf.position(startPos);
            }
            default -> throw new IllegalArgumentException("unsupported CT2 dtype: " + dtypeId);
        }
        buf.position(startPos + numBytes);
        return data;
    }

    private static String readString(final ByteBuffer buf) {
        int len = buf.getShort() & 0xFFFF;
        byte[] bytes = new byte[len];
        buf.get(bytes);
        // CT2 strings are null-terminated
        int strLen = len > 0 && bytes[len - 1] == 0 ? len - 1 : len;
        return new String(bytes, 0, strLen, StandardCharsets.UTF_8);
    }
}

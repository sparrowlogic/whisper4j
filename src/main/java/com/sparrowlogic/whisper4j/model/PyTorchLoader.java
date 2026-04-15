package com.sparrowlogic.whisper4j.model;

import com.sparrowlogic.whisper4j.tensor.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.*;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Loads Whisper model weights from PyTorch .pt files.
 * PyTorch .pt files are ZIP archives containing pickle-serialized state dicts.
 * This implements a minimal pickle interpreter sufficient for Whisper checkpoints.
 */
public final class PyTorchLoader implements ModelLoader {

    private static final Logger LOG = Logger.getLogger(PyTorchLoader.class.getName());

    @Override
    public WeightStore load(Path path) throws IOException {
        LOG.info("Loading PyTorch model from " + path);
        WeightStore store = new WeightStore();
        Map<String, byte[]> dataFiles = new HashMap<>();

        try (var zip = new ZipFile(path.toFile())) {
            // first pass: collect raw tensor data files
            var entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                String name = entry.getName();
                if (name.endsWith(".pkl")) continue;
                dataFiles.put(name, zip.getInputStream(entry).readAllBytes());
            }

            // second pass: parse pickle to get tensor metadata
            entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                if (!entry.getName().endsWith(".pkl")) continue;
                byte[] pkl = zip.getInputStream(entry).readAllBytes();
                parsePkl(pkl, dataFiles, store, entry.getName());
            }
        }
        return store;
    }

    /**
     * Minimal pickle protocol parser for PyTorch state_dict.
     * Handles only the opcodes used by torch.save() for Whisper models.
     */
    private void parsePkl(byte[] pkl, Map<String, byte[]> dataFiles, WeightStore store, String pklName) {
        var stack = new ArrayDeque<Object>();
        var memo = new HashMap<Integer, Object>();
        var markStack = new ArrayDeque<Integer>();
        int pos = 0;

        // strip archive prefix from pkl name for data file resolution
        String archivePrefix = pklName.contains("/")
                ? pklName.substring(0, pklName.lastIndexOf('/') + 1) : "";

        while (pos < pkl.length) {
            int op = pkl[pos++] & 0xFF;
            switch (op) {
                case 0x80 -> pos++; // PROTO
                case '}' -> stack.push(new LinkedHashMap<>()); // EMPTY_DICT
                case ']' -> stack.push(new ArrayList<>()); // EMPTY_LIST
                case ')' -> stack.push(new ArrayList<>()); // EMPTY_TUPLE
                case '(' -> markStack.push(stack.size()); // MARK
                case 0x85 -> { // TUPLE1
                    Object a = stack.pop();
                    stack.push(List.of(a));
                }
                case 0x86 -> { // TUPLE2
                    Object b = stack.pop(), a = stack.pop();
                    stack.push(List.of(a, b));
                }
                case 0x87 -> { // TUPLE3
                    Object c = stack.pop(), b = stack.pop(), a = stack.pop();
                    stack.push(List.of(a, b, c));
                }
                case 't' -> { // TUPLE (from mark)
                    int mark = markStack.pop();
                    List<Object> items = popToMark(stack, mark);
                    stack.push(items);
                }
                case 'l' -> { // LIST (from mark)
                    int mark = markStack.pop();
                    stack.push(popToMark(stack, mark));
                }
                case 'u' -> { // SETITEMS
                    int mark = markStack.pop();
                    List<Object> items = popToMark(stack, mark);
                    @SuppressWarnings("unchecked")
                    var dict = (Map<Object, Object>) stack.peek();
                    for (int i = 0; i < items.size(); i += 2) {
                        dict.put(items.get(i), items.get(i + 1));
                    }
                }
                case 's' -> { // SETITEM
                    Object val = stack.pop(), key = stack.pop();
                    @SuppressWarnings("unchecked")
                    var dict = (Map<Object, Object>) stack.peek();
                    dict.put(key, val);
                }
                case 'e' -> { // APPENDS
                    int mark = markStack.pop();
                    List<Object> items = popToMark(stack, mark);
                    @SuppressWarnings("unchecked")
                    var list = (List<Object>) stack.peek();
                    list.addAll(items);
                }
                case 'a' -> { // APPEND
                    Object item = stack.pop();
                    @SuppressWarnings("unchecked")
                    var list = (List<Object>) stack.peek();
                    list.add(item);
                }
                case 0x8C -> { // SHORT_BINUNICODE
                    int len = pkl[pos++] & 0xFF;
                    stack.push(new String(pkl, pos, len));
                    pos += len;
                }
                case 'X' -> { // BINUNICODE
                    int len = readInt32(pkl, pos); pos += 4;
                    stack.push(new String(pkl, pos, len));
                    pos += len;
                }
                case 'c' -> { // GLOBAL
                    int nl1 = indexOf(pkl, pos, '\n');
                    int nl2 = indexOf(pkl, nl1 + 1, '\n');
                    String module = new String(pkl, pos, nl1 - pos);
                    String name = new String(pkl, nl1 + 1, nl2 - nl1 - 1);
                    stack.push(new Global(module, name));
                    pos = nl2 + 1;
                }
                case 0x8E -> { // LONG_BINGET
                    int idx = readInt32(pkl, pos); pos += 4;
                    stack.push(memo.get(idx));
                }
                case 'h' -> { // BINGET
                    int idx = pkl[pos++] & 0xFF;
                    stack.push(memo.get(idx));
                }
                case 'q' -> { // BINPUT
                    int idx = pkl[pos++] & 0xFF;
                    memo.put(idx, stack.peek());
                }
                case 'r' -> { // LONG_BINPUT
                    int idx = readInt32(pkl, pos); pos += 4;
                    memo.put(idx, stack.peek());
                }
                case 'R' -> { // REDUCE: call callable with args
                    Object args = stack.pop();
                    Object callable = stack.pop();
                    stack.push(reduce(callable, args, archivePrefix, dataFiles));
                }
                case 'b' -> { // BUILD: obj.__setstate__(arg)
                    Object arg = stack.pop();
                    Object obj = stack.peek();
                    build(obj, arg);
                }
                case 0x88 -> stack.push(true); // NEWTRUE
                case 0x89 -> stack.push(false); // NEWFALSE
                case 'N' -> stack.push(null); // NONE
                case 'J' -> { stack.push(readInt32(pkl, pos)); pos += 4; } // BININT
                case 'K' -> stack.push(pkl[pos++] & 0xFF); // BININT1
                case 'M' -> { stack.push(readUInt16(pkl, pos)); pos += 2; } // BININT2
                case 0x8A -> { // LONG1
                    int n = pkl[pos++] & 0xFF;
                    long val = 0;
                    for (int i = 0; i < n; i++) val |= (long)(pkl[pos++] & 0xFF) << (8 * i);
                    stack.push(val);
                }
                case 'G' -> { // BINFLOAT
                    long bits = 0;
                    for (int i = 0; i < 8; i++) bits = (bits << 8) | (pkl[pos++] & 0xFF);
                    stack.push(Double.longBitsToDouble(bits));
                }
                case 'Q' -> { // BINPERSID
                    Object pid = stack.pop();
                    stack.push(resolvePersId(pid, archivePrefix, dataFiles));
                }
                case '.' -> { // STOP
                    if (!stack.isEmpty() && stack.peek() instanceof Map<?,?> topDict) {
                        extractWeights(topDict, store);
                    }
                    return;
                }
                case 0x95 -> pos += 8; // FRAME
                case 0x94 -> { // MEMOIZE
                    memo.put(memo.size(), stack.peek());
                }
                case 0x93 -> { // STACK_GLOBAL
                    Object name = stack.pop();
                    Object module = stack.pop();
                    stack.push(new Global(module.toString(), name.toString()));
                }
                case 0x91 -> { // NEWOBJ_EX — pop kwargs, args, cls, push placeholder
                    stack.pop(); stack.pop();
                    Object cls = stack.pop();
                    stack.push(new Rebuild(cls));
                }
                case 0x81 -> { // NEWOBJ — pop args, cls, push placeholder
                    Object args = stack.pop();
                    Object cls = stack.pop();
                    stack.push(reduce(cls, args, archivePrefix, dataFiles));
                }
                default -> throw new IllegalStateException("unknown pickle opcode: 0x%02X at pos %d".formatted(op, pos - 1));
            }
        }
        // extract model_state_dict from top of stack
        if (!stack.isEmpty() && stack.peek() instanceof Map<?,?> topDict) {
            extractWeights(topDict, store);
        }
    }

    @SuppressWarnings("unchecked")
    private void extractWeights(Map<?, ?> topDict, WeightStore store) {
        // Whisper .pt has {"dims": {...}, "model_state_dict": {...}} or flat state_dict
        Object stateDict = topDict.containsKey("model_state_dict") ? topDict.get("model_state_dict") : topDict;
        if (stateDict instanceof Map<?,?> sd) {
            for (var entry : sd.entrySet()) {
                String name = entry.getKey().toString();
                Object val = entry.getValue();
                if (val instanceof TensorData td) {
                    store.put(name, td.toTensor());
                } else if (val instanceof Rebuild rb && rb.data != null) {
                    store.put(name, rb.data.toTensor());
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    private Object reduce(Object callable, Object args, String prefix, Map<String, byte[]> dataFiles) {
        if (callable instanceof Global g) {
            if ("torch._utils".equals(g.module) && "_rebuild_tensor_v2".equals(g.name)) {
                // args = (storage, offset, shape, stride, requires_grad, OrderedDict)
                if (args instanceof List<?> argList && argList.size() >= 3
                        && argList.get(0) instanceof TensorStorage storage) {
                    int offset = argList.get(1) instanceof Number n ? n.intValue() : 0;
                    List<?> shape = argList.get(2) instanceof List<?> s ? s : List.of();
                    int[] dims = shape.stream().mapToInt(o -> ((Number) o).intValue()).toArray();
                    Rebuild rb = new Rebuild(g);
                    rb.data = new TensorData(storage, offset, dims);
                    return rb;
                }
                return new Rebuild(g);
            }
            if ("collections".equals(g.module) && "OrderedDict".equals(g.name)) {
                return new LinkedHashMap<>();
            }
            if ("torch".equals(g.module)) {
                return new Rebuild(g); // dtype markers like torch.float32
            }
        }
        return new Rebuild(callable);
    }

    @SuppressWarnings("unchecked")
    private void build(Object obj, Object arg) {
        if (obj instanceof Rebuild rb && arg instanceof List<?> args) {
            // _rebuild_tensor_v2(storage, offset, shape, stride)
            if (args.size() >= 4 && args.get(0) instanceof TensorStorage storage) {
                List<?> shape = args.get(2) instanceof List<?> s ? s : List.of();
                int[] dims = shape.stream().mapToInt(o -> ((Number) o).intValue()).toArray();
                int offset = args.get(1) instanceof Number n ? n.intValue() : 0;
                rb.data = new TensorData(storage, offset, dims);
            }
        }
        if (obj instanceof Map && arg instanceof Map) {
            ((Map<Object,Object>) obj).putAll((Map<?,?>) arg);
        }
    }

    private Object resolvePersId(Object pid, String prefix, Map<String, byte[]> dataFiles) {
        if (pid instanceof List<?> parts && parts.size() >= 5) {
            // ("storage", storageType, key, location, numElements)
            String key = parts.get(2).toString();
            int numElements = ((Number) parts.get(4)).intValue();
            String dataPath = prefix + "data/" + key;
            byte[] raw = dataFiles.get(dataPath);
            if (raw == null) {
                // try without prefix
                for (var entry : dataFiles.entrySet()) {
                    if (entry.getKey().endsWith("data/" + key)) {
                        raw = entry.getValue();
                        break;
                    }
                }
            }
            return new TensorStorage(raw, numElements, dtypeFromGlobal(parts.get(1)));
        }
        return null;
    }

    private String dtypeFromGlobal(Object obj) {
        if (obj instanceof Global g) return g.name; // e.g. "FloatStorage", "HalfStorage"
        return "FloatStorage";
    }

    // ---- Internal data types ----

    private record Global(String module, String name) {}

    private static class Rebuild {
        Object source;
        TensorData data;
        Rebuild(Object source) { this.source = source; }
    }

    private record TensorStorage(byte[] raw, int numElements, String storageType) {}

    private record TensorData(TensorStorage storage, int offset, int[] shape) {
        Tensor toTensor() {
            int size = 1;
            for (int d : shape) size *= d;
            float[] data = new float[size];
            ByteBuffer buf = ByteBuffer.wrap(storage.raw).order(ByteOrder.LITTLE_ENDIAN);
            switch (storage.storageType) {
                case "FloatStorage" -> {
                    buf.position(offset * 4);
                    buf.asFloatBuffer().get(data, 0, size);
                }
                case "HalfStorage" -> {
                    buf.position(offset * 2);
                    for (int i = 0; i < size; i++) data[i] = Float.float16ToFloat(buf.getShort());
                }
                case "BFloat16Storage" -> {
                    buf.position(offset * 2);
                    for (int i = 0; i < size; i++) data[i] = Float.intBitsToFloat((buf.getShort() & 0xFFFF) << 16);
                }
                default -> throw new IllegalArgumentException("unsupported storage: " + storage.storageType);
            }
            return Tensor.of(data, shape);
        }
    }

    // ---- Helpers ----

    private static int readInt32(byte[] b, int off) {
        return (b[off] & 0xFF) | ((b[off+1] & 0xFF) << 8) | ((b[off+2] & 0xFF) << 16) | ((b[off+3] & 0xFF) << 24);
    }

    private static int readUInt16(byte[] b, int off) {
        return (b[off] & 0xFF) | ((b[off+1] & 0xFF) << 8);
    }

    private static int indexOf(byte[] b, int from, char c) {
        for (int i = from; i < b.length; i++) if (b[i] == c) return i;
        return b.length;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> popToMark(Deque<Object> stack, int mark) {
        List<Object> items = new ArrayList<>();
        while (stack.size() > mark) items.addFirst(stack.pop());
        return items;
    }
}

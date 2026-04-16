package com.sparrowlogic.whisper4j;

import org.jspecify.annotations.Nullable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Builder and registry for {@link WhisperModel} instances with LRU eviction.
 *
 * <h3>Single model:</h3>
 * <pre>{@code
 * var model = new WhisperModelFactory(Path.of("ggml-base.en.bin"))
 *         .language("en")
 *         .beamSize(5)
 *         .create();
 * }</pre>
 *
 * <h3>Named shared instances with eviction:</h3>
 * <pre>{@code
 * WhisperModelFactory.register("preview", new WhisperModelFactory(Path.of("ggml-tiny.en.bin"))
 *         .beamSize(1));
 * WhisperModelFactory.register("final", new WhisperModelFactory(Path.of("ggml-base.en.bin"))
 *         .beamSize(5));
 *
 * WhisperModelFactory.shared("preview").transcribe(audio);
 *
 * // Explicit eviction when no longer needed
 * WhisperModelFactory.evict("preview");
 *
 * // Or LRU eviction under memory pressure — keep only the 2 most recently used
 * WhisperModelFactory.evictLeastRecentlyUsed(2);
 * }</pre>
 *
 * <h3>Spring Boot:</h3>
 * <pre>{@code
 * @Bean
 * public WhisperModel whisperModel(@Value("${whisper.model-path}") String path) {
 *     return new WhisperModelFactory(Path.of(path))
 *             .language("en")
 *             .beamSize(5)
 *             .create();
 * }
 * }</pre>
 */
public final class WhisperModelFactory {

    private static final Logger LOG = Logger.getLogger(WhisperModelFactory.class.getName());
    private static final ConcurrentHashMap<String, WhisperModelFactory> FACTORIES = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, CachedModel> SHARED = new ConcurrentHashMap<>();

    private final Path modelPath;
    private @Nullable Path tokenizerPath;
    private String language = "en";
    private String task = "transcribe";
    private int beamSize = 5;
    private boolean withTimestamps = true;
    private boolean vadFilter;
    private boolean conditionOnPrevious;

    public WhisperModelFactory(final Path modelPath) {
        this.modelPath = modelPath;
    }

    // ---- Builder methods ----

    public WhisperModelFactory tokenizerPath(final Path path) {
        this.tokenizerPath = path;
        return this;
    }

    public WhisperModelFactory language(final String lang) {
        this.language = lang;
        return this;
    }

    public WhisperModelFactory task(final String t) {
        this.task = t;
        return this;
    }

    public WhisperModelFactory beamSize(final int size) {
        this.beamSize = size;
        return this;
    }

    public WhisperModelFactory withTimestamps(final boolean enabled) {
        this.withTimestamps = enabled;
        return this;
    }

    public WhisperModelFactory vadFilter(final boolean enabled) {
        this.vadFilter = enabled;
        return this;
    }

    public WhisperModelFactory conditionOnPrevious(final boolean enabled) {
        this.conditionOnPrevious = enabled;
        return this;
    }

    /** Build {@link WhisperModel.TranscriptionOptions} from this factory's settings. */
    public WhisperModel.TranscriptionOptions buildOptions() {
        return new WhisperModel.TranscriptionOptions(
                this.language, this.task, this.beamSize,
                this.withTimestamps, this.vadFilter, this.conditionOnPrevious);
    }

    /**
     * Load and return a new {@link WhisperModel}.
     * @throws IOException if the model file cannot be read or is corrupt
     */
    public WhisperModel create() throws IOException {
        return WhisperModel.load(this.modelPath, this.tokenizerPath, this.buildOptions());
    }

    // ---- Static shared registry with LRU eviction ----

    /** Register a named factory. The model is loaded lazily on first {@link #shared} call. */
    public static void register(final String name, final WhisperModelFactory factory) {
        FACTORIES.put(name, factory);
    }

    /**
     * Get a shared model instance by name. Loads lazily on first access (thread-safe).
     * @throws IllegalArgumentException if no factory is registered for the name
     * @throws UncheckedIOException if the model fails to load
     */
    public static WhisperModel shared(final String name) {
        CachedModel cached = SHARED.computeIfAbsent(name, k -> {
            WhisperModelFactory factory = FACTORIES.get(k);
            if (factory == null) {
                throw new IllegalArgumentException("No model registered with name: " + k);
            }
            try {
                return new CachedModel(factory.create());
            } catch (final IOException e) {
                throw new UncheckedIOException("Failed to load model '" + k + "'", e);
            }
        });
        cached.touch();
        return cached.get();
    }

    /**
     * Evict a named model, releasing its reference for garbage collection.
     * The factory remains registered — the model will be reloaded on next {@link #shared} call.
     * Safe to call while other threads are using the model (they hold their own reference).
     */
    public static void evict(final String name) {
        CachedModel removed = SHARED.remove(name);
        if (removed != null) {
            removed.get().close();
            LOG.info("Evicted model: " + name);
        }
    }

    /**
     * Evict least-recently-used models until at most {@code maxEntries} remain loaded.
     * Models are evicted in order of last access time (oldest first).
     * Registered factories are preserved — evicted models reload on next access.
     *
     * @param maxEntries maximum number of loaded models to keep
     * @return number of models evicted
     */
    public static int evictLeastRecentlyUsed(final int maxEntries) {
        int toEvict = SHARED.size() - maxEntries;
        if (toEvict <= 0) {
            return 0;
        }
        // Sort by last access time, evict oldest
        var entries = SHARED.entrySet().stream()
                .sorted(Map.Entry.comparingByValue())
                .limit(toEvict)
                .toList();
        int evicted = 0;
        for (final var entry : entries) {
            if (SHARED.remove(entry.getKey(), entry.getValue())) {
                entry.getValue().get().close();
                LOG.info("LRU evicted model: " + entry.getKey());
                evicted++;
            }
        }
        return evicted;
    }

    /** Check if a name is registered. */
    public static boolean isRegistered(final String name) {
        return FACTORIES.containsKey(name);
    }

    /** Number of currently loaded (not just registered) models. */
    public static int loadedCount() {
        return SHARED.size();
    }

    /** Clear all registered factories and close all cached models. */
    public static void clearRegistry() {
        SHARED.values().forEach(cached -> cached.get().close());
        FACTORIES.clear();
        SHARED.clear();
    }

    /** Wrapper that tracks last-access time for LRU eviction. */
    private static final class CachedModel implements Comparable<CachedModel> {
        private final WhisperModel model;
        private volatile long lastAccessNanos;

        CachedModel(final WhisperModel model) {
            this.model = model;
            this.lastAccessNanos = System.nanoTime();
        }

        WhisperModel get() {
            return this.model;
        }

        void touch() {
            this.lastAccessNanos = System.nanoTime();
        }

        @Override
        public int compareTo(final CachedModel other) {
            return Long.compare(this.lastAccessNanos, other.lastAccessNanos);
        }
    }
}

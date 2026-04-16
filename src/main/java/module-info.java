module com.sparrowlogic.whisper4j {
    requires java.logging;
    requires static jdk.incubator.vector; // optional: Java 26+ Vector API SIMD acceleration
    exports com.sparrowlogic.whisper4j;
    exports com.sparrowlogic.whisper4j.annotation;
    exports com.sparrowlogic.whisper4j.tensor;
    exports com.sparrowlogic.whisper4j.audio;
    exports com.sparrowlogic.whisper4j.tokenizer;
    exports com.sparrowlogic.whisper4j.model;
    exports com.sparrowlogic.whisper4j.nn;
}

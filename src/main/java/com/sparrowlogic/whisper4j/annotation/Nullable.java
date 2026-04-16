package com.sparrowlogic.whisper4j.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Indicates that the annotated element may be {@code null}.
 * Local replacement for {@code org.jspecify.annotations.Nullable} to avoid
 * JPMS module resolution issues with multi-release JAR compilation.
 */
@Documented
@Retention(RetentionPolicy.CLASS)
@Target(ElementType.TYPE_USE)
public @interface Nullable {
}

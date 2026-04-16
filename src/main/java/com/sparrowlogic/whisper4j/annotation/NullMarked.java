package com.sparrowlogic.whisper4j.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Indicates that the annotated scope assumes non-null by default.
 * Local replacement for {@code org.jspecify.annotations.NullMarked} to avoid
 * JPMS module resolution issues with multi-release JAR compilation.
 */
@Documented
@Retention(RetentionPolicy.CLASS)
@Target({ElementType.PACKAGE, ElementType.TYPE, ElementType.METHOD, ElementType.CONSTRUCTOR})
public @interface NullMarked {
}

/**
 * Pure Java implementation of OpenAI Whisper speech-to-text.
 *
 * <p>Main entry points:
 * <ul>
 *   <li>{@link com.sparrowlogic.whisper4j.WhisperModel} — load models and transcribe audio</li>
 *   <li>{@link com.sparrowlogic.whisper4j.WhisperModelFactory} — builder/registry for DI frameworks</li>
 *   <li>{@link com.sparrowlogic.whisper4j.Alignment} — DTW word-level timestamp alignment</li>
 * </ul>
 */
@NullMarked
package com.sparrowlogic.whisper4j;

import org.jspecify.annotations.NullMarked;

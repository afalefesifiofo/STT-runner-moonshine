// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0
//
// bindings/java/MoonshineRunner.java
//
// JNI wrapper for MoonshineRunner — for Android and desktop Java/Kotlin apps.
//
// Usage in Android (build.gradle):
//   externalNativeBuild { cmake { path "CMakeLists.txt" } }
//   defaultConfig { ndk { abiFilters "arm64-v8a" } }

package com.moonshine.runner;

/**
 * JNI wrapper around the native MoonshineRunner C++ library.
 *
 * <p>Call {@link #create(String, String, int, boolean)} to obtain an instance,
 * then {@link #transcribe(float[])} for inference.
 *
 * <p>Thread-safety: A single instance must not be called from multiple threads
 * concurrently. Use one instance per thread, or synchronize externally.
 */
public class MoonshineRunner implements AutoCloseable {

    static {
        System.loadLibrary("moonshine_runner_jni");
    }

    // Native handle (pointer to RunnerImpl)
    private long nativeHandle = 0;

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Create a new MoonshineRunner.
     *
     * @param modelDir      Path to directory containing *.onnx + tokenizer.json
     * @param modelSize     "tiny" or "base"
     * @param threads       Number of intra-op threads (recommend 4)
     * @param useArmXnnpack Enable Arm XNNPACK execution provider
     * @return              Initialized MoonshineRunner
     * @throws RuntimeException if initialization fails
     */
    public static MoonshineRunner create(
            String modelDir,
            String modelSize,
            int threads,
            boolean useArmXnnpack) {
        MoonshineRunner r = new MoonshineRunner();
        r.nativeHandle = nativeCreate(modelDir, modelSize, threads, useArmXnnpack);
        if (r.nativeHandle == 0) {
            throw new RuntimeException("Failed to create native MoonshineRunner. Check model files.");
        }
        return r;
    }

    /** Convenience factory with defaults (tiny model, 4 threads). */
    public static MoonshineRunner create(String modelDir) {
        return create(modelDir, "tiny", 4, false);
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /**
     * Transcribe a segment of raw audio.
     *
     * @param pcm float32 PCM samples at 16 kHz, mono, range [-1, 1]
     * @return    Transcript text
     */
    public String transcribe(float[] pcm) {
        if (nativeHandle == 0) throw new IllegalStateException("Runner is closed");
        return nativeTranscribe(nativeHandle, pcm);
    }

    /** Returns the ORT execution provider string (e.g. "XNNPACK" or "CPU"). */
    public String getExecutionProvider() {
        if (nativeHandle == 0) return "closed";
        return nativeGetEP(nativeHandle);
    }

    /** Sample rate expected by the model (always 16000 Hz). */
    public static int getSampleRate() { return 16000; }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    @Override
    public void close() {
        if (nativeHandle != 0) {
            nativeDestroy(nativeHandle);
            nativeHandle = 0;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }

    // ------------------------------------------------------------------
    // Native declarations
    // ------------------------------------------------------------------

    private static native long   nativeCreate(String modelDir, String modelSize,
                                              int threads, boolean useXnnpack);
    private static native String nativeTranscribe(long handle, float[] pcm);
    private static native String nativeGetEP(long handle);
    private static native void   nativeDestroy(long handle);
}

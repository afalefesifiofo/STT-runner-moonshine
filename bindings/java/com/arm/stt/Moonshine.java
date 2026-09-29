// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0

package com.arm.stt;

import com.moonshine.runner.MoonshineRunner;

/**
 * Adapter to maintain API compatibility with the old STT-Runner interface,
 * delegating under the hood to the new MoonshineRunner.
 */
public class Moonshine {

    // Keep track of the active runner instance
    private static MoonshineRunner activeRunner = null;
    private static String cachedModelPath = null;

    // A dummy configuration class to match old API
    public static class MoonshineConfig {
        public int threads = 4;
        // Whisper-specific params will be ignored by Moonshine
        public String prompt = ""; 
    }

    /**
     * Replaces the old initContext.
     * returns a dummy context pointer (e.g., 1L).
     */
    public static long initContext(String modelPath, String sharedLibraryPath) {
        // MoonshineRunner manages its own ORT loading.
        cachedModelPath = modelPath;
        return 1L; // Dummy context
    }

    /**
     * Initializes parameters. Creates the actual MoonshineRunner here since we need threads.
     */
    public static void initParameters(MoonshineConfig config) {
        if (activeRunner != null) {
            activeRunner.close();
        }
        
        // We assume "tiny" by default for this adapter unless the path suggests otherwise
        String size = cachedModelPath != null && cachedModelPath.contains("base") ? "base" : "tiny";
        
        // Use the cached model path or a default
        String path = cachedModelPath != null ? cachedModelPath : "models/moonshine-tiny";
        
        activeRunner = MoonshineRunner.create(path, size, config.threads, true);
    }

    /**
     * Transcribe using the active runner.
     */
    public static String fullTranscribe(long context, float[] audio) {
        if (activeRunner == null) {
            throw new IllegalStateException("MoonshineRunner not initialized. Call initParameters first.");
        }
        return activeRunner.transcribe(audio);
    }

    /**
     * Cleanup resources.
     */
    public static void freeContext(long context) {
        if (activeRunner != null) {
            activeRunner.close();
            activeRunner = null;
        }
    }
}

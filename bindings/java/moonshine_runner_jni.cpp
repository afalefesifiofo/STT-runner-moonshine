// SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
// SPDX-License-Identifier: Apache-2.0

#include <jni.h>
#include <string>
#include <memory>
#include "moonshine_runner/runner.hpp"

// Convert jstring to std::string
static std::string jstring2string(JNIEnv* env, jstring jStr) {
    if (!jStr) return "";
    const char* cstr = env->GetStringUTFChars(jStr, nullptr);
    std::string str(cstr);
    env->ReleaseStringUTFChars(jStr, cstr);
    return str;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_moonshine_runner_MoonshineRunner_nativeCreate(
    JNIEnv* env, jclass /* clazz */,
    jstring jModelDir, jstring jModelSize, jint threads, jboolean useXnnpack) 
{
    try {
        moonshine::RunnerConfig config;
        config.model_dir = jstring2string(env, jModelDir);
        
        std::string size_str = jstring2string(env, jModelSize);
        if (size_str == "base") {
            config.model_size = moonshine::ModelSize::Base;
        } else {
            config.model_size = moonshine::ModelSize::Tiny;
        }
        
        config.intra_op_threads = threads;
        config.use_arm_xnnpack = useXnnpack;

        auto* runner = new moonshine::MoonshineRunner(config);
        return reinterpret_cast<jlong>(runner);
    } catch (const std::exception& e) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_moonshine_runner_MoonshineRunner_nativeTranscribe(
    JNIEnv* env, jobject /* thiz */, jlong handle, jfloatArray jPcm) 
{
    auto* runner = reinterpret_cast<moonshine::MoonshineRunner*>(handle);
    if (!runner || !jPcm) return env->NewStringUTF("");

    jsize len = env->GetArrayLength(jPcm);
    jfloat* pcm = env->GetFloatArrayElements(jPcm, nullptr);

    try {
        auto result = runner->transcribe(pcm, len);
        env->ReleaseFloatArrayElements(jPcm, pcm, JNI_ABORT);
        return env->NewStringUTF(result.text.c_str());
    } catch (const std::exception& e) {
        env->ReleaseFloatArrayElements(jPcm, pcm, JNI_ABORT);
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, e.what());
        return env->NewStringUTF("");
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_moonshine_runner_MoonshineRunner_nativeGetEP(
    JNIEnv* env, jobject /* thiz */, jlong handle) 
{
    auto* runner = reinterpret_cast<moonshine::MoonshineRunner*>(handle);
    if (!runner) return env->NewStringUTF("");

    return env->NewStringUTF(runner->execution_provider().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_moonshine_runner_MoonshineRunner_nativeDestroy(
    JNIEnv* /* env */, jobject /* thiz */, jlong handle) 
{
    auto* runner = reinterpret_cast<moonshine::MoonshineRunner*>(handle);
    delete runner;
}

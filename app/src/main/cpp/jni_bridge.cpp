#include <jni.h>
#include <string>
#include <memory>
#include "flux_engine.h"

static std::unique_ptr<flux::FluxEngine> gEngine;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeInit(
    JNIEnv* env,
    jobject /* this */,
    jstring jModelDir) {
    if (!jModelDir) return JNI_FALSE;
    const char* modelDirStr = env->GetStringUTFChars(jModelDir, nullptr);
    if (!gEngine) {
        gEngine = std::make_unique<flux::FluxEngine>();
    }
    bool success = gEngine->initialize(modelDirStr);
    env->ReleaseStringUTFChars(jModelDir, modelDirStr);
    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeRelease(
    JNIEnv* /* env */,
    jobject /* this */) {
    if (gEngine) {
        gEngine->release();
        gEngine.reset();
    }
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeGenerateNoise(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jOutLatent,
    jint numElements,
    jlong seed) {
    if (!gEngine || !jOutLatent) return JNI_FALSE;
    jfloat* ptr = env->GetFloatArrayElements(jOutLatent, nullptr);
    bool ok = gEngine->generateGaussianNoise(ptr, numElements, static_cast<uint64_t>(seed));
    env->ReleaseFloatArrayElements(jOutLatent, ptr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeInterpolateInitLatent(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jOutNoisyLatent,
    jfloatArray jEncodedLatent,
    jfloatArray jNoise,
    jint numElements,
    jfloat initStrength) {
    if (!gEngine || !jOutNoisyLatent || !jEncodedLatent || !jNoise) return JNI_FALSE;
    jfloat* outPtr = env->GetFloatArrayElements(jOutNoisyLatent, nullptr);
    jfloat* encPtr = env->GetFloatArrayElements(jEncodedLatent, nullptr);
    jfloat* noisePtr = env->GetFloatArrayElements(jNoise, nullptr);

    bool ok = gEngine->interpolateInitLatent(outPtr, encPtr, noisePtr, numElements, initStrength);

    env->ReleaseFloatArrayElements(jOutNoisyLatent, outPtr, 0);
    env->ReleaseFloatArrayElements(jEncodedLatent, encPtr, JNI_ABORT);
    env->ReleaseFloatArrayElements(jNoise, noisePtr, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativePatchifyLatent(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jVaeLatent,
    jfloatArray jPackedTokens,
    jint height,
    jint width) {
    if (!gEngine || !jVaeLatent || !jPackedTokens) return JNI_FALSE;
    jfloat* vaePtr = env->GetFloatArrayElements(jVaeLatent, nullptr);
    jfloat* packedPtr = env->GetFloatArrayElements(jPackedTokens, nullptr);

    bool ok = gEngine->patchifyLatent(vaePtr, packedPtr, height, width);

    env->ReleaseFloatArrayElements(jVaeLatent, vaePtr, JNI_ABORT);
    env->ReleaseFloatArrayElements(jPackedTokens, packedPtr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeUnpatchifyLatent(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jPackedTokens,
    jfloatArray jVaeLatent,
    jint height,
    jint width) {
    if (!gEngine || !jPackedTokens || !jVaeLatent) return JNI_FALSE;
    jfloat* packedPtr = env->GetFloatArrayElements(jPackedTokens, nullptr);
    jfloat* vaePtr = env->GetFloatArrayElements(jVaeLatent, nullptr);

    bool ok = gEngine->unpatchifyLatent(packedPtr, vaePtr, height, width);

    env->ReleaseFloatArrayElements(jPackedTokens, packedPtr, JNI_ABORT);
    env->ReleaseFloatArrayElements(jVaeLatent, vaePtr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeComputeRoPEImage(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jOutRotary,
    jint height,
    jint width,
    jint hiddenDim) {
    if (!gEngine || !jOutRotary) return JNI_FALSE;
    jfloat* ptr = env->GetFloatArrayElements(jOutRotary, nullptr);
    bool ok = gEngine->computeRoPEImage(ptr, height, width, hiddenDim);
    env->ReleaseFloatArrayElements(jOutRotary, ptr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeComputeRoPEText(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jOutRotary,
    jint seqLen,
    jint hiddenDim) {
    if (!gEngine || !jOutRotary) return JNI_FALSE;
    jfloat* ptr = env->GetFloatArrayElements(jOutRotary, nullptr);
    bool ok = gEngine->computeRoPEText(ptr, seqLen, hiddenDim);
    env->ReleaseFloatArrayElements(jOutRotary, ptr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeComputeJointRoPE(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jOutCos,
    jfloatArray jOutSin,
    jint height,
    jint width,
    jint textSeqLen,
    jint numCondImages) {
    if (!gEngine || !jOutCos || !jOutSin) return JNI_FALSE;
    jfloat* cosPtr = env->GetFloatArrayElements(jOutCos, nullptr);
    jfloat* sinPtr = env->GetFloatArrayElements(jOutSin, nullptr);
    bool ok = gEngine->computeJointRoPE(cosPtr, sinPtr, height, width, textSeqLen, numCondImages);
    env->ReleaseFloatArrayElements(jOutCos, cosPtr, 0);
    env->ReleaseFloatArrayElements(jOutSin, sinPtr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeReadTimeGuidanceEmbed(
    JNIEnv* env,
    jobject /* this */,
    jint stepIdx,
    jfloatArray jOutEmbed) {
    if (!gEngine || !jOutEmbed) return JNI_FALSE;
    jfloat* ptr = env->GetFloatArrayElements(jOutEmbed, nullptr);
    bool ok = gEngine->readTimeGuidanceEmbed(stepIdx, ptr);
    env->ReleaseFloatArrayElements(jOutEmbed, ptr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeComputeTimeGuidanceEmbed(
    JNIEnv* env,
    jobject /* this */,
    jfloat timestep1000,
    jfloatArray jOutEmbed) {
    if (!gEngine || !jOutEmbed) return JNI_FALSE;
    jfloat* ptr = env->GetFloatArrayElements(jOutEmbed, nullptr);
    bool ok = gEngine->computeTimeGuidanceEmbed(timestep1000, ptr);
    env->ReleaseFloatArrayElements(jOutEmbed, ptr, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeEulerStep(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jNoisyLatent,
    jfloatArray jVelocity,
    jint numElements,
    jfloat dt) {
    if (!gEngine || !jNoisyLatent || !jVelocity) return JNI_FALSE;
    jfloat* noisyPtr = env->GetFloatArrayElements(jNoisyLatent, nullptr);
    jfloat* velPtr = env->GetFloatArrayElements(jVelocity, nullptr);

    bool ok = gEngine->eulerStep(noisyPtr, velPtr, numElements, dt);

    env->ReleaseFloatArrayElements(jNoisyLatent, noisyPtr, 0);
    env->ReleaseFloatArrayElements(jVelocity, velPtr, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_gestura_fluxedit_engine_FluxNativeEngine_nativeValidateFinite(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jData) {
    if (!jData) return JNI_FALSE;
    jsize len = env->GetArrayLength(jData);
    jfloat* ptr = env->GetFloatArrayElements(jData, nullptr);
    bool ok = flux::FluxEngine::validateFinite(ptr, static_cast<size_t>(len));
    env->ReleaseFloatArrayElements(jData, ptr, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"

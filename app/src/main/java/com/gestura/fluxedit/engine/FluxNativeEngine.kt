package com.gestura.fluxedit.engine

import android.util.Log

object FluxNativeEngine {
    private const val TAG = "FluxNativeEngine"
    private var isLoaded = false

    init {
        try {
            System.loadLibrary("flux_engine")
            isLoaded = true
            Log.i(TAG, "libflux_engine.so successfully loaded")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Could not load libflux_engine.so: ${e.message}")
            isLoaded = false
        }
    }

    fun isNativeLoaded(): Boolean = isLoaded

    external fun nativeInit(modelDir: String): Boolean
    external fun nativeRelease()
    external fun nativeGenerateNoise(outLatent: FloatArray, numElements: Int, seed: Long): Boolean
    external fun nativeInterpolateInitLatent(
        outNoisyLatent: FloatArray,
        encodedLatent: FloatArray,
        noise: FloatArray,
        numElements: Int,
        initStrength: Float
    ): Boolean
    external fun nativePatchifyLatent(
        vaeLatent: FloatArray,
        packedTokens: FloatArray,
        height: Int,
        width: Int
    ): Boolean
    external fun nativeUnpatchifyLatent(
        packedTokens: FloatArray,
        vaeLatent: FloatArray,
        height: Int,
        width: Int
    ): Boolean
    external fun nativeComputeRoPEImage(
        outRotary: FloatArray,
        height: Int,
        width: Int,
        hiddenDim: Int = 128
    ): Boolean
    external fun nativeComputeRoPEText(
        outRotary: FloatArray,
        seqLen: Int = 512,
        hiddenDim: Int = 128
    ): Boolean
    external fun nativeComputeJointRoPE(
        outCos: FloatArray,
        outSin: FloatArray,
        height: Int,
        width: Int,
        textSeqLen: Int = 512,
        numCondImages: Int = 0
    ): Boolean
    external fun nativeReadTimeGuidanceEmbed(stepIdx: Int, outEmbed: FloatArray): Boolean
    external fun nativeComputeTimeGuidanceEmbed(timestep1000: Float, outEmbed: FloatArray): Boolean
    external fun nativeEulerStep(
        noisyLatent: FloatArray,
        velocity: FloatArray,
        numElements: Int,
        dt: Float
    ): Boolean
    external fun nativeValidateFinite(data: FloatArray): Boolean
}

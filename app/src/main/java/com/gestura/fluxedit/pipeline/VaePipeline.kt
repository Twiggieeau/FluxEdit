package com.gestura.fluxedit.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import com.gestura.fluxedit.engine.FluxNativeEngine
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VaePipeline(private val modelDir: File) {
    private val TAG = "VaePipeline"

    fun init() {
        // Models are loaded on-demand and closed immediately after each encode/decode call
    }

    fun close() {
        // Nothing held open persistently
    }

    private fun runVaeStage(
        fileName: String,
        signatureName: String,
        inputArray: FloatArray,
        outputElements: Int
    ): FloatArray {
        val modelFile = File(modelDir, fileName)
        if (!modelFile.exists()) {
            throw IllegalStateException("Missing VAE model file: ${modelFile.absolutePath}")
        }

        // Primary execution via LiteRT CompiledModel (only initializes requested geometry signature)
        var model: CompiledModel? = null
        var inputBuffers: Map<String, TensorBuffer>? = null
        var outputBuffers: Map<String, TensorBuffer>? = null
        try {
            model = CompiledModel.create(
                modelFile.absolutePath,
                CompiledModel.Options(Accelerator.CPU)
            )
            inputBuffers = model.createInputBuffersBySignature(signatureName)
            outputBuffers = model.createOutputBuffersBySignature(signatureName)

            inputBuffers["args_0"]?.writeFloat(inputArray)
                ?: throw IllegalStateException("$fileName missing input 'args_0'")

            model.run(inputBuffers, outputBuffers, signatureName)

            val outBuf = outputBuffers["output_0"]
                ?: throw IllegalStateException("$fileName missing output 'output_0'")
            return outBuf.readFloat()
        } catch (litertErr: Throwable) {
            Log.w(TAG, "CompiledModel fallback to Interpreter for $fileName ($signatureName): ${litertErr.message}")
        } finally {
            inputBuffers?.values?.forEach { try { it.close() } catch (_: Throwable) {} }
            outputBuffers?.values?.forEach { try { it.close() } catch (_: Throwable) {} }
            try { model?.close() } catch (_: Throwable) {}
        }

        // Fallback to single-use TFLite Interpreter (closed immediately in finally)
        val options = Interpreter.Options().apply {
            numThreads = 4
        }
        var interpreter: Interpreter? = null
        try {
            interpreter = Interpreter(modelFile, options)
            val inputByteBuffer = ByteBuffer.allocateDirect(inputArray.size * 4).order(ByteOrder.nativeOrder())
            inputByteBuffer.asFloatBuffer().put(inputArray)
            inputByteBuffer.rewind()

            val outputByteBuffer = ByteBuffer.allocateDirect(outputElements * 4).order(ByteOrder.nativeOrder())
            val inputs = mapOf("args_0" to inputByteBuffer)
            val outputs = mapOf("output_0" to outputByteBuffer)

            try {
                interpreter.runSignature(inputs, outputs, signatureName)
            } catch (e: Exception) {
                Log.w(TAG, "Signature $signatureName execution fallback: ${e.message}")
                inputByteBuffer.rewind()
                outputByteBuffer.rewind()
                val legacyInputs = arrayOf<Any>(inputByteBuffer)
                val legacyOutputs = mutableMapOf<Int, Any>(0 to outputByteBuffer)
                interpreter.runForMultipleInputsOutputs(legacyInputs, legacyOutputs)
            }

            outputByteBuffer.rewind()
            val out = FloatArray(outputElements)
            outputByteBuffer.asFloatBuffer().get(out)
            return out
        } finally {
            try { interpreter?.close() } catch (_: Throwable) {}
        }
    }

    /**
     * Encodes an input Bitmap to a packed latent tensor [numTokens, 128]
     * using the exact geometry SignatureDef.
     */
    fun encodeImageToPackedLatent(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): FloatArray {
        // 1. Resize image to target geometry
        val scaled = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)

        // 2 & 3. Planar RGB [-1.0, 1.0] [1, 3, H, W]
        val planeSize = targetHeight * targetWidth
        val pixels = IntArray(planeSize)
        scaled.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        val inputPlanar = FloatArray(3 * planeSize)
        for (i in 0 until planeSize) {
            val px = pixels[i]
            inputPlanar[i] = (Color.red(px) / 127.5f) - 1.0f
            inputPlanar[planeSize + i] = (Color.green(px) / 127.5f) - 1.0f
            inputPlanar[2 * planeSize + i] = (Color.blue(px) / 127.5f) - 1.0f
        }

        // 4. Output buffer for [1, 32, H/8, W/8]
        val hLatent = targetHeight / 8
        val wLatent = targetWidth / 8
        val latentElements = 32 * hLatent * wLatent
        val signatureName = "geometry_${targetWidth}x${targetHeight}"

        val vaeLatent = runVaeStage("kv_vae_enc.tflite", signatureName, inputPlanar, latentElements)

        // 5. Patchify & normalize via AutoencoderKLFlux2 BatchNorm into width-128 tokens
        val numTokens = (targetHeight / 16) * (targetWidth / 16)
        val packedTokens = FloatArray(numTokens * 128)
        FluxNativeEngine.nativePatchifyLatent(vaeLatent, packedTokens, targetHeight, targetWidth)

        return packedTokens
    }

    /**
     * Decodes a packed latent tensor [numTokens, 128] back to an RGB Bitmap.
     */
    fun decodePackedLatentToBitmap(packedTokens: FloatArray, width: Int, height: Int): Bitmap? {
        // 1. Denormalize via AutoencoderKLFlux2 BatchNorm & unpack width-128 latent to [1, 32, H/8, W/8]
        val hLatent = height / 8
        val wLatent = width / 8
        val latentElements = 32 * hLatent * wLatent
        val vaeLatent = FloatArray(latentElements)
        FluxNativeEngine.nativeUnpatchifyLatent(packedTokens, vaeLatent, height, width)

        // 2. Planar RGB output [1, 3, H, W]
        val planeSize = height * width
        val signatureName = "geometry_${width}x${height}"
        val planarRgb = runVaeStage("kv_vae.tflite", signatureName, vaeLatent, 3 * planeSize)

        // 3. Reject NaN/Inf and map [-1, 1] to RGB Bitmap
        if (!FluxNativeEngine.nativeValidateFinite(planarRgb)) {
            Log.e(TAG, "Decoded image contains non-finite values (NaN / Inf)!")
            return null
        }

        val outPixels = IntArray(planeSize)
        for (i in 0 until planeSize) {
            val rVal = planarRgb[i]
            val gVal = planarRgb[planeSize + i]
            val bVal = planarRgb[2 * planeSize + i]

            val r = ((rVal + 1.0f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((gVal + 1.0f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((bVal + 1.0f) * 127.5f).toInt().coerceIn(0, 255)

            outPixels[i] = Color.rgb(r, g, b)
        }

        val outBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        outBitmap.setPixels(outPixels, 0, width, 0, 0, width, height)
        return outBitmap
    }
}

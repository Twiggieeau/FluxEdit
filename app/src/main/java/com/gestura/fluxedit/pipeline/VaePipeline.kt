package com.gestura.fluxedit.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import com.gestura.fluxedit.engine.FluxNativeEngine
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VaePipeline(private val modelDir: File) {
    private val TAG = "VaePipeline"

    private var encInterpreter: Interpreter? = null
    private var decInterpreter: Interpreter? = null

    fun init() {
        val encFile = File(modelDir, "kv_vae_enc.tflite")
        val decFile = File(modelDir, "kv_vae.tflite")

        val options = Interpreter.Options().apply {
            numThreads = 4
        }

        if (encFile.exists()) encInterpreter = Interpreter(encFile, options)
        if (decFile.exists()) decInterpreter = Interpreter(decFile, options)
    }

    fun close() {
        encInterpreter?.close()
        decInterpreter?.close()
    }

    /**
     * Encodes an input Bitmap to a packed latent tensor [numTokens, 128]
     * using the exact geometry SignatureDef and direct ByteBuffers.
     */
    fun encodeImageToPackedLatent(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): FloatArray {
        // 1. Resize image to target geometry
        val scaled = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)

        // 2 & 3. Planar RGB [-1.0, 1.0] in a direct ByteBuffer to preserve 4D tensor shape [1, 3, H, W]
        val planeSize = targetHeight * targetWidth
        val pixels = IntArray(planeSize)
        scaled.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        val inputByteBuffer = ByteBuffer.allocateDirect(1 * 3 * planeSize * 4).order(ByteOrder.nativeOrder())
        
        // Channel R
        for (i in 0 until planeSize) {
            inputByteBuffer.putFloat((Color.red(pixels[i]) / 127.5f) - 1.0f)
        }
        // Channel G
        for (i in 0 until planeSize) {
            inputByteBuffer.putFloat((Color.green(pixels[i]) / 127.5f) - 1.0f)
        }
        // Channel B
        for (i in 0 until planeSize) {
            inputByteBuffer.putFloat((Color.blue(pixels[i]) / 127.5f) - 1.0f)
        }
        inputByteBuffer.rewind()

        // 4. Output buffer for [1, 32, H/8, W/8]
        val hLatent = targetHeight / 8
        val wLatent = targetWidth / 8
        val latentElements = 32 * hLatent * wLatent
        val outputByteBuffer = ByteBuffer.allocateDirect(latentElements * 4).order(ByteOrder.nativeOrder())

        // 5. Invoke using SignatureDef "geometry_{W}x{H}"
        val interpreter = encInterpreter
        if (interpreter != null) {
            val signatureName = "geometry_${targetWidth}x${targetHeight}"
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
        }

        // 6. Extract VAE latent float array
        outputByteBuffer.rewind()
        val vaeLatent = FloatArray(latentElements)
        for (i in 0 until latentElements) {
            vaeLatent[i] = outputByteBuffer.getFloat()
        }

        // 7. Patchify & pack into width-128 tokens
        val numTokens = (targetHeight / 16) * (targetWidth / 16)
        val packedTokens = FloatArray(numTokens * 128)
        FluxNativeEngine.nativePatchifyLatent(vaeLatent, packedTokens, targetHeight, targetWidth)

        return packedTokens
    }

    /**
     * Decodes a packed latent tensor [numTokens, 128] back to an RGB Bitmap.
     */
    fun decodePackedLatentToBitmap(packedTokens: FloatArray, width: Int, height: Int): Bitmap? {
        // 1. Unpack width-128 latent to [1, 32, H/8, W/8]
        val hLatent = height / 8
        val wLatent = width / 8
        val latentElements = 32 * hLatent * wLatent
        val vaeLatent = FloatArray(latentElements)
        FluxNativeEngine.nativeUnpatchifyLatent(packedTokens, vaeLatent, height, width)

        val inputByteBuffer = ByteBuffer.allocateDirect(latentElements * 4).order(ByteOrder.nativeOrder())
        for (v in vaeLatent) {
            inputByteBuffer.putFloat(v)
        }
        inputByteBuffer.rewind()

        // 2. Planar RGB output buffer [1, 3, H, W]
        val planeSize = height * width
        val outputByteBuffer = ByteBuffer.allocateDirect(1 * 3 * planeSize * 4).order(ByteOrder.nativeOrder())

        val interpreter = decInterpreter ?: return null
        val signatureName = "geometry_${width}x${height}"
        val inputs = mapOf("args_0" to inputByteBuffer)
        val outputs = mapOf("output_0" to outputByteBuffer)

        try {
            interpreter.runSignature(inputs, outputs, signatureName)
        } catch (e: Exception) {
            Log.w(TAG, "Decoder signature $signatureName execution fallback: ${e.message}")
            inputByteBuffer.rewind()
            outputByteBuffer.rewind()
            val legacyInputs = arrayOf<Any>(inputByteBuffer)
            val legacyOutputs = mutableMapOf<Int, Any>(0 to outputByteBuffer)
            interpreter.runForMultipleInputsOutputs(legacyInputs, legacyOutputs)
        }

        outputByteBuffer.rewind()
        val planarRgb = FloatArray(3 * planeSize)
        for (i in 0 until 3 * planeSize) {
            planarRgb[i] = outputByteBuffer.getFloat()
        }

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

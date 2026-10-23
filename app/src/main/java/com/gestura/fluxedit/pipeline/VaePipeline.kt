package com.gestura.fluxedit.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import com.gestura.fluxedit.engine.FluxNativeEngine
import org.tensorflow.lite.Interpreter
import java.io.File

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
     * Encodes an input Bitmap to a packed latent tensor [numTokens, 128].
     */
    fun encodeImageToPackedLatent(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): FloatArray {
        // 1. Resize image to exact geometry
        val scaled = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)

        // 2 & 3 & 4. Convert to planar RGB [-1.0, 1.0]
        // Planar format: [1, 3, H, W]
        val planarRgb = FloatArray(3 * targetHeight * targetWidth)
        val pixels = IntArray(targetWidth * targetHeight)
        scaled.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        val planeSize = targetHeight * targetWidth
        for (i in 0 until planeSize) {
            val color = pixels[i]
            val r = Color.red(color)
            val g = Color.green(color)
            val b = Color.blue(color)

            planarRgb[i] = (r / 127.5f) - 1.0f
            planarRgb[planeSize + i] = (g / 127.5f) - 1.0f
            planarRgb[2 * planeSize + i] = (b / 127.5f) - 1.0f
        }

        // 5. Run VAE encoder: produces [1, 32, H/8, W/8] latent
        val hLatent = targetHeight / 8
        val wLatent = targetWidth / 8
        val vaeLatent = FloatArray(32 * hLatent * wLatent)

        val interpreter = encInterpreter
        if (interpreter != null) {
            val inputs = arrayOf<Any>(planarRgb)
            val outputs = mutableMapOf<Int, Any>(0 to vaeLatent)
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
        }

        // 6. Patchify & pack into width-128 tokens
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
        val vaeLatent = FloatArray(32 * hLatent * wLatent)
        FluxNativeEngine.nativeUnpatchifyLatent(packedTokens, vaeLatent, height, width)

        // 2 & 3. Invoke kv_vae.tflite -> planar RGB [-1.0, 1.0]
        val planeSize = height * width
        val planarRgb = FloatArray(3 * planeSize)

        val interpreter = decInterpreter
        if (interpreter != null) {
            val inputs = arrayOf<Any>(vaeLatent)
            val outputs = mutableMapOf<Int, Any>(0 to planarRgb)
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
        } else {
            return null
        }

        // 4 & 5. Reject NaN or Infinity, map [-1, 1] to RGB pixel range
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

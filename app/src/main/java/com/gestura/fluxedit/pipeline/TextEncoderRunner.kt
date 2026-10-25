package com.gestura.fluxedit.pipeline

import android.util.Log
import com.gestura.fluxedit.engine.FluxNativeEngine
import com.gestura.fluxedit.tokenizer.TokenizedPrompt
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TextEncoderRunner(private val modelDir: File) {
    private val TAG = "TextEncoderRunner"

    private var embedTableChannel: FileChannel? = null
    private var embedRaf: RandomAccessFile? = null

    companion object {
        const val SEQ_LEN = 512
        const val HIDDEN_DIM = 2560
        const val PROMPT_COND_DIM = 7680 // 3 * 2560 taps composed
        const val BYTES_PER_FP16_ROW = HIDDEN_DIM * 2
    }

    fun init() {
        if (embedTableChannel != null) return
        val embedFile = File(modelDir, "tokenizer/qwen_embed_fp16.bin")
        if (embedFile.exists()) {
            embedRaf = RandomAccessFile(embedFile, "r")
            embedTableChannel = embedRaf?.channel
        }
    }

    fun close() {
        try {
            embedTableChannel?.close()
        } catch (_: Exception) {}
        try {
            embedRaf?.close()
        } catch (_: Exception) {}
        embedTableChannel = null
        embedRaf = null
    }

    fun lookupEmbeddingsFloatArray(tokenIds: IntArray): FloatArray {
        init()
        val out = FloatArray(1 * SEQ_LEN * HIDDEN_DIM)
        val channel = embedTableChannel ?: return out

        val rowBuffer = ByteBuffer.allocateDirect(BYTES_PER_FP16_ROW).order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until SEQ_LEN) {
            val tokenId = tokenIds[i]
            val offset = tokenId.toLong() * BYTES_PER_FP16_ROW
            rowBuffer.clear()
            channel.read(rowBuffer, offset)
            rowBuffer.flip()

            val baseIdx = i * HIDDEN_DIM
            for (j in 0 until HIDDEN_DIM) {
                val halfBits = rowBuffer.short.toInt() and 0xFFFF
                out[baseIdx + j] = fp16ToFp32(halfBits)
            }
        }
        return out
    }

    /**
     * Runs a single 923 MB Qwen3 text encoder stage and immediately closes its native model
     * so two 923 MB stages are never resident in memory simultaneously.
     */
    private fun runSingleEncoderStage(
        fileName: String,
        hiddenIn: FloatArray,
        maskArray: FloatArray,
        cosArray: FloatArray,
        sinArray: FloatArray
    ): FloatArray {
        val stageFile = File(modelDir, fileName)
        if (!stageFile.exists()) {
            throw IllegalStateException("Missing text encoder stage file: ${stageFile.absolutePath}")
        }
        val sigName = "text_512"
        val outElements = 1 * SEQ_LEN * HIDDEN_DIM

        // Primary execution via LiteRT CompiledModel (only initializes the requested signature)
        var model: CompiledModel? = null
        val inputBuffers = HashMap<String, TensorBuffer>(4)
        val outputBuffers = HashMap<String, TensorBuffer>(1)
        try {
            model = CompiledModel.create(
                stageFile.absolutePath,
                CompiledModel.Options(Accelerator.CPU)
            )
            inputBuffers["args_0"] = model.createInputBuffer(sigName, "args_0").also { it.writeFloat(hiddenIn) }
            inputBuffers["args_1"] = model.createInputBuffer(sigName, "args_1").also { it.writeFloat(maskArray) }
            inputBuffers["args_2"] = model.createInputBuffer(sigName, "args_2").also { it.writeFloat(cosArray) }
            inputBuffers["args_3"] = model.createInputBuffer(sigName, "args_3").also { it.writeFloat(sinArray) }

            val outBuf = model.createOutputBuffer(sigName, "output_0")
            outputBuffers["output_0"] = outBuf

            model.run(inputBuffers, outputBuffers, sigName)
            return outBuf.readFloat()
        } catch (litertErr: Throwable) {
            Log.w(TAG, "CompiledModel fallback to Interpreter for $fileName: ${litertErr.message}")
        } finally {
            inputBuffers.values.forEach { try { it.close() } catch (_: Throwable) {} }
            outputBuffers.values.forEach { try { it.close() } catch (_: Throwable) {} }
            try { model?.close() } catch (_: Throwable) {}
        }

        // Fallback to single-stage TFLite Interpreter (closed immediately in finally)
        val options = Interpreter.Options().apply {
            numThreads = 4
        }
        var interp: Interpreter? = null
        try {
            interp = Interpreter(stageFile, options)
            val in0 = floatArrayToDirectBuffer(hiddenIn)
            val in1 = floatArrayToDirectBuffer(maskArray)
            val in2 = floatArrayToDirectBuffer(cosArray)
            val in3 = floatArrayToDirectBuffer(sinArray)
            val out0 = ByteBuffer.allocateDirect(outElements * 4).order(ByteOrder.nativeOrder())

            val inputs = mapOf(
                "args_0" to in0,
                "args_1" to in1,
                "args_2" to in2,
                "args_3" to in3
            )
            val outputs = mapOf(
                "output_0" to out0
            )
            interp.runSignature(inputs, outputs, sigName)
            out0.rewind()
            val result = FloatArray(outElements)
            out0.asFloatBuffer().get(result)
            return result
        } finally {
            try { interp?.close() } catch (_: Throwable) {}
        }
    }

    private fun floatArrayToDirectBuffer(arr: FloatArray): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(arr.size * 4).order(ByteOrder.nativeOrder())
        buf.asFloatBuffer().put(arr)
        buf.rewind()
        return buf
    }

    fun encode(tokenized: TokenizedPrompt): FloatArray {
        // 1. Embedding lookup [1, 512, 2560]
        val embedArray = lookupEmbeddingsFloatArray(tokenized.tokenIds)

        // 2. 4D Attention mask [1, 32, 512, 512]
        val maskArray = FloatArray(1 * 32 * SEQ_LEN * SEQ_LEN)
        val activeLen = tokenized.unpaddedLength
        val headStride = SEQ_LEN * SEQ_LEN
        for (q in 0 until SEQ_LEN) {
            val rowOffset = q * SEQ_LEN
            for (k in 0 until SEQ_LEN) {
                val isVisible = (k < activeLen && k <= q)
                maskArray[rowOffset + k] = if (isVisible) 0.0f else -10000.0f
            }
        }
        for (head in 1 until 32) {
            System.arraycopy(maskArray, 0, maskArray, head * headStride, headStride)
        }

        // 3. RoPE tensors for Qwen3 text: cos [512, 128] and sin [512, 128]
        // Qwen3 rotate_half slices [..., 64:128] and [..., 0:64], so each 128-dim row is [freqs_0..63, freqs_0..63]
        val ropeCombined = FloatArray(SEQ_LEN * 128)
        FluxNativeEngine.nativeComputeRoPEText(ropeCombined, SEQ_LEN, 128)

        val cosArray = FloatArray(SEQ_LEN * 128)
        val sinArray = FloatArray(SEQ_LEN * 128)

        for (pos in 0 until SEQ_LEN) {
            val baseIdx = pos * 128
            for (d in 0 until 64) {
                val c = ropeCombined[baseIdx + d * 2]
                val s = ropeCombined[baseIdx + d * 2 + 1]
                cosArray[baseIdx + d] = c
                sinArray[baseIdx + d] = s
                cosArray[baseIdx + 64 + d] = c
                sinArray[baseIdx + 64 + d] = s
            }
        }

        // Execute Stage 0, Stage 1, Stage 2 strictly sequentially (closing each 923MB model before loading the next)
        val tap0 = runSingleEncoderStage("ke_enc0.tflite", embedArray, maskArray, cosArray, sinArray)
        val tap1 = runSingleEncoderStage("ke_enc1.tflite", tap0, maskArray, cosArray, sinArray)
        val tap2 = runSingleEncoderStage("ke_enc2.tflite", tap1, maskArray, cosArray, sinArray)

        // Compose 7,680-dim prompt conditioning by concatenating [tap0, tap1, tap2] per token
        val promptConditioning = FloatArray(SEQ_LEN * PROMPT_COND_DIM)
        for (i in 0 until SEQ_LEN) {
            val srcOffset = i * HIDDEN_DIM
            val condOffset = i * PROMPT_COND_DIM
            System.arraycopy(tap0, srcOffset, promptConditioning, condOffset, HIDDEN_DIM)
            System.arraycopy(tap1, srcOffset, promptConditioning, condOffset + HIDDEN_DIM, HIDDEN_DIM)
            System.arraycopy(tap2, srcOffset, promptConditioning, condOffset + 2 * HIDDEN_DIM, HIDDEN_DIM)
        }

        return promptConditioning
    }

    private fun fp16ToFp32(h: Int): Float {
        val sign = (h ushr 15) and 0x0001
        var exp = (h ushr 10) and 0x001f
        var frac = h and 0x03ff

        val res = when (exp) {
            0 -> {
                if (frac == 0) {
                    sign shl 31
                } else {
                    exp = 1
                    while ((frac and 0x0400) == 0) {
                        frac = frac shl 1
                        exp--
                    }
                    frac = frac and 0x03ff
                    (sign shl 31) or ((exp + 112) shl 23) or (frac shl 13)
                }
            }
            31 -> (sign shl 31) or 0x7f800000 or (frac shl 13)
            else -> (sign shl 31) or ((exp + 112) shl 23) or (frac shl 13)
        }
        return java.lang.Float.intBitsToFloat(res)
    }
}

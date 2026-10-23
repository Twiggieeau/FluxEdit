package com.gestura.fluxedit.pipeline

import android.util.Log
import com.gestura.fluxedit.engine.FluxNativeEngine
import com.gestura.fluxedit.tokenizer.TokenizedPrompt
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TextEncoderRunner(private val modelDir: File) {
    private val TAG = "TextEncoderRunner"

    private var interp0: Interpreter? = null
    private var interp1: Interpreter? = null
    private var interp2: Interpreter? = null

    private var embedTableChannel: FileChannel? = null
    private var embedRaf: RandomAccessFile? = null

    companion object {
        const val SEQ_LEN = 512
        const val HIDDEN_DIM = 2560
        const val PROMPT_COND_DIM = 7680 // 3 * 2560 taps composed
        const val BYTES_PER_FP16_ROW = HIDDEN_DIM * 2
    }

    fun init() {
        val f0 = File(modelDir, "ke_enc0.tflite")
        val f1 = File(modelDir, "ke_enc1.tflite")
        val f2 = File(modelDir, "ke_enc2.tflite")

        val options = Interpreter.Options().apply {
            numThreads = 4
        }

        if (f0.exists()) interp0 = Interpreter(f0, options)
        if (f1.exists()) interp1 = Interpreter(f1, options)
        if (f2.exists()) interp2 = Interpreter(f2, options)

        val embedFile = File(modelDir, "tokenizer/qwen_embed_fp16.bin")
        if (embedFile.exists()) {
            embedRaf = RandomAccessFile(embedFile, "r")
            embedTableChannel = embedRaf?.channel
        }
    }

    fun close() {
        interp0?.close()
        interp1?.close()
        interp2?.close()
        embedTableChannel?.close()
        embedRaf?.close()
    }

    /**
     * Reads embedding table row for each of the 512 tokens.
     * Output shape: [1, 512, 2560] float32.
     */
    fun lookupEmbeddings(tokenIds: IntArray): FloatArray {
        val embeddings = FloatArray(SEQ_LEN * HIDDEN_DIM)
        val channel = embedTableChannel ?: return embeddings

        val rowBuffer = ByteBuffer.allocateDirect(BYTES_PER_FP16_ROW).order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until SEQ_LEN) {
            val tokenId = tokenIds[i]
            val offset = tokenId.toLong() * BYTES_PER_FP16_ROW
            rowBuffer.clear()
            channel.read(rowBuffer, offset)
            rowBuffer.flip()

            val rowOffset = i * HIDDEN_DIM
            for (j in 0 until HIDDEN_DIM) {
                val halfBits = rowBuffer.short.toInt() and 0xFFFF
                embeddings[rowOffset + j] = fp16ToFp32(halfBits)
            }
        }
        return embeddings
    }

    /**
     * Executes the 3-stage text encoder and composes the 7,680-dim prompt conditioning.
     */
    fun encode(tokenized: TokenizedPrompt): FloatArray {
        val promptConditioning = FloatArray(SEQ_LEN * PROMPT_COND_DIM)

        // 1. Embedding lookup
        val embedInput = lookupEmbeddings(tokenized.tokenIds)

        // 2. Rotary position tensor for text
        val textRoPE = FloatArray(SEQ_LEN * 128)
        FluxNativeEngine.nativeComputeRoPEText(textRoPE, SEQ_LEN, 128)

        // 3. Stage 0: ke_enc0.tflite
        val tap0 = FloatArray(SEQ_LEN * HIDDEN_DIM)
        val intermediate0 = FloatArray(SEQ_LEN * HIDDEN_DIM)
        runStage0(embedInput, tokenized.attentionMask, textRoPE, intermediate0, tap0)

        // 4. Stage 1: ke_enc1.tflite
        val tap1 = FloatArray(SEQ_LEN * HIDDEN_DIM)
        val intermediate1 = FloatArray(SEQ_LEN * HIDDEN_DIM)
        runStage1(intermediate0, tokenized.attentionMask, textRoPE, intermediate1, tap1)

        // 5. Stage 2: ke_enc2.tflite
        val tap2 = FloatArray(SEQ_LEN * HIDDEN_DIM)
        runStage2(intermediate1, tokenized.attentionMask, textRoPE, tap2)

        // 6. Compose the 7,680-dimensional prompt conditioning per token: [tap0, tap1, tap2]
        for (i in 0 until SEQ_LEN) {
            val condOffset = i * PROMPT_COND_DIM
            val tapOffset = i * HIDDEN_DIM

            System.arraycopy(tap0, tapOffset, promptConditioning, condOffset, HIDDEN_DIM)
            System.arraycopy(tap1, tapOffset, promptConditioning, condOffset + HIDDEN_DIM, HIDDEN_DIM)
            System.arraycopy(tap2, tapOffset, promptConditioning, condOffset + 2 * HIDDEN_DIM, HIDDEN_DIM)
        }

        // 7. Validate output shapes and finite state
        if (!FluxNativeEngine.nativeValidateFinite(promptConditioning)) {
            Log.e(TAG, "Prompt conditioning contains non-finite values (NaN / Inf)!")
        }

        return promptConditioning
    }

    private fun runStage0(
        embeds: FloatArray,
        attnMask: FloatArray,
        rope: FloatArray,
        outNext: FloatArray,
        outTap: FloatArray
    ) {
        val interpreter = interp0 ?: return
        val inputs = arrayOf<Any>(embeds, attnMask, rope)
        val outputs = mutableMapOf<Int, Any>(
            0 to outNext,
            1 to outTap
        )
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
    }

    private fun runStage1(
        intermediate: FloatArray,
        attnMask: FloatArray,
        rope: FloatArray,
        outNext: FloatArray,
        outTap: FloatArray
    ) {
        val interpreter = interp1 ?: return
        val inputs = arrayOf<Any>(intermediate, attnMask, rope)
        val outputs = mutableMapOf<Int, Any>(
            0 to outNext,
            1 to outTap
        )
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
    }

    private fun runStage2(
        intermediate: FloatArray,
        attnMask: FloatArray,
        rope: FloatArray,
        outTap: FloatArray
    ) {
        val interpreter = interp2 ?: return
        val inputs = arrayOf<Any>(intermediate, attnMask, rope)
        val outputs = mutableMapOf<Int, Any>(
            0 to outTap
        )
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
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

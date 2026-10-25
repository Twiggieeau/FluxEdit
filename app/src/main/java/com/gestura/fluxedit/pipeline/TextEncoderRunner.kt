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

    fun lookupEmbeddings(tokenIds: IntArray): ByteBuffer {
        val embedBuffer = ByteBuffer.allocateDirect(1 * SEQ_LEN * HIDDEN_DIM * 4).order(ByteOrder.nativeOrder())
        val channel = embedTableChannel ?: return embedBuffer

        val rowBuffer = ByteBuffer.allocateDirect(BYTES_PER_FP16_ROW).order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until SEQ_LEN) {
            val tokenId = tokenIds[i]
            val offset = tokenId.toLong() * BYTES_PER_FP16_ROW
            rowBuffer.clear()
            channel.read(rowBuffer, offset)
            rowBuffer.flip()

            for (j in 0 until HIDDEN_DIM) {
                val halfBits = rowBuffer.short.toInt() and 0xFFFF
                embedBuffer.putFloat(fp16ToFp32(halfBits))
            }
        }
        embedBuffer.rewind()
        return embedBuffer
    }

    fun encode(tokenized: TokenizedPrompt): FloatArray {
        val promptConditioning = FloatArray(SEQ_LEN * PROMPT_COND_DIM)

        // 1. Embedding lookup [1, 512, 2560]
        val embedBuffer = lookupEmbeddings(tokenized.tokenIds)

        // 2. 4D Attention mask [1, 32, 512, 512] in direct ByteBuffer
        val maskBuffer = ByteBuffer.allocateDirect(1 * 32 * SEQ_LEN * SEQ_LEN * 4).order(ByteOrder.nativeOrder())
        val activeLen = tokenized.unpaddedLength
        for (head in 0 until 32) {
            for (q in 0 until SEQ_LEN) {
                for (k in 0 until SEQ_LEN) {
                    val isVisible = (k < activeLen && k <= q)
                    maskBuffer.putFloat(if (isVisible) 0.0f else -10000.0f)
                }
            }
        }
        maskBuffer.rewind()

        // 3. RoPE tensors for Qwen3 text: cos [512, 128] and sin [512, 128]
        // Qwen3 rotate_half slices [..., 64:128] and [..., 0:64], so each 128-dim row is [freqs_0..63, freqs_0..63]
        val ropeCombined = FloatArray(SEQ_LEN * 128)
        FluxNativeEngine.nativeComputeRoPEText(ropeCombined, SEQ_LEN, 128)

        val cosBuffer = ByteBuffer.allocateDirect(SEQ_LEN * 128 * 4).order(ByteOrder.nativeOrder())
        val sinBuffer = ByteBuffer.allocateDirect(SEQ_LEN * 128 * 4).order(ByteOrder.nativeOrder())

        for (pos in 0 until SEQ_LEN) {
            val baseIdx = pos * 128
            // First half (0..63)
            for (d in 0 until 64) {
                cosBuffer.putFloat(ropeCombined[baseIdx + d * 2])
                sinBuffer.putFloat(ropeCombined[baseIdx + d * 2 + 1])
            }
            // Second half (64..127) duplicates (0..63) for rotate_half pairing
            for (d in 0 until 64) {
                cosBuffer.putFloat(ropeCombined[baseIdx + d * 2])
                sinBuffer.putFloat(ropeCombined[baseIdx + d * 2 + 1])
            }
        }
        cosBuffer.rewind()
        sinBuffer.rewind()

        // Buffers for the 3 hidden state taps (layers 9, 18, 27), each [1, 512, 2560]
        val tap0Buffer = ByteBuffer.allocateDirect(1 * SEQ_LEN * HIDDEN_DIM * 4).order(ByteOrder.nativeOrder())
        val tap1Buffer = ByteBuffer.allocateDirect(1 * SEQ_LEN * HIDDEN_DIM * 4).order(ByteOrder.nativeOrder())
        val tap2Buffer = ByteBuffer.allocateDirect(1 * SEQ_LEN * HIDDEN_DIM * 4).order(ByteOrder.nativeOrder())

        val sigName = "text_512"

        // Stage 0 (layers 0..8 -> outputs hidden_states[9] as output_0)
        interp0?.let { interp ->
            val inputs = mapOf(
                "args_0" to embedBuffer,
                "args_1" to maskBuffer,
                "args_2" to cosBuffer,
                "args_3" to sinBuffer
            )
            val outputs = mapOf(
                "output_0" to tap0Buffer
            )
            try {
                interp.runSignature(inputs, outputs, sigName)
            } catch (e: Exception) {
                Log.e(TAG, "Stage 0 execution failed: ${e.message}", e)
            }
        }

        // Stage 1 (layers 9..17 -> takes tap0Buffer, outputs hidden_states[18] as output_0)
        tap0Buffer.rewind()
        maskBuffer.rewind()
        cosBuffer.rewind()
        sinBuffer.rewind()
        interp1?.let { interp ->
            val inputs = mapOf(
                "args_0" to tap0Buffer,
                "args_1" to maskBuffer,
                "args_2" to cosBuffer,
                "args_3" to sinBuffer
            )
            val outputs = mapOf(
                "output_0" to tap1Buffer
            )
            try {
                interp.runSignature(inputs, outputs, sigName)
            } catch (e: Exception) {
                Log.e(TAG, "Stage 1 execution failed: ${e.message}", e)
            }
        }

        // Stage 2 (layers 18..26 -> takes tap1Buffer, outputs hidden_states[27] as output_0)
        tap1Buffer.rewind()
        maskBuffer.rewind()
        cosBuffer.rewind()
        sinBuffer.rewind()
        interp2?.let { interp ->
            val inputs = mapOf(
                "args_0" to tap1Buffer,
                "args_1" to maskBuffer,
                "args_2" to cosBuffer,
                "args_3" to sinBuffer
            )
            val outputs = mapOf(
                "output_0" to tap2Buffer
            )
            try {
                interp.runSignature(inputs, outputs, sigName)
            } catch (e: Exception) {
                Log.e(TAG, "Stage 2 execution failed: ${e.message}", e)
            }
        }

        // Compose 7,680-dim prompt conditioning by concatenating [tap0, tap1, tap2] per token
        tap0Buffer.rewind()
        tap1Buffer.rewind()
        tap2Buffer.rewind()

        for (i in 0 until SEQ_LEN) {
            val condOffset = i * PROMPT_COND_DIM
            for (j in 0 until HIDDEN_DIM) {
                promptConditioning[condOffset + j] = if (tap0Buffer.hasRemaining()) tap0Buffer.getFloat() else 0f
            }
            for (j in 0 until HIDDEN_DIM) {
                promptConditioning[condOffset + HIDDEN_DIM + j] = if (tap1Buffer.hasRemaining()) tap1Buffer.getFloat() else 0f
            }
            for (j in 0 until HIDDEN_DIM) {
                promptConditioning[condOffset + 2 * HIDDEN_DIM + j] = if (tap2Buffer.hasRemaining()) tap2Buffer.getFloat() else 0f
            }
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

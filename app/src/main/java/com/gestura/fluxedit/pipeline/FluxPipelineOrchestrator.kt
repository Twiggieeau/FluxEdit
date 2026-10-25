package com.gestura.fluxedit.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.gestura.fluxedit.engine.FluxNativeEngine
import com.gestura.fluxedit.model.ModelManifest
import com.gestura.fluxedit.tokenizer.QwenBpeTokenizer
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.exp

enum class WorkflowType {
    SINGLE_IMAGE_EDIT,
    TWO_IMAGE_EDIT,
    TEXT_TO_IMAGE
}

data class GenerationParams(
    val prompt: String,
    val workflow: WorkflowType,
    val image1: Bitmap? = null,
    val image2: Bitmap? = null,
    val width: Int = 512,
    val height: Int = 512,
    val seed: Long = System.currentTimeMillis(),
    val initStrength: Float = 1.0f // 1.0 = full FLUX.2 Klein reference conditioning; < 1.0 = latent blend
)

sealed class PipelineProgress {
    object Idle : PipelineProgress()
    object Tokenizing : PipelineProgress()
    object EncodingText : PipelineProgress()
    data class EncodingImages(val current: Int, val total: Int) : PipelineProgress()
    data class Denoising(
        val step: Int,
        val totalSteps: Int = 4,
        val percent: Float,
        val stageName: String = ""
    ) : PipelineProgress()
    object DecodingVae : PipelineProgress()
    data class Completed(val outputBitmap: Bitmap, val durationMs: Long) : PipelineProgress()
    data class Error(val message: String) : PipelineProgress()
}

class FluxPipelineOrchestrator(private val context: Context) {
    private val TAG = "FluxPipelineOrch"

    private val modelDir: File by lazy {
        File(context.filesDir, "flux2-klein-4b")
    }

    private val tokenizer = QwenBpeTokenizer()
    private var textEncoder: TextEncoderRunner? = null
    private var vaePipeline: VaePipeline? = null
    private var isInitialized = false

    companion object {
        const val TEXT_SEQ_LEN = 512
        const val HIDDEN_DIM = 3072
        const val LATENT_CHANNELS = 128
        const val ROPE_DIM = 64

        /**
         * Computes the FLUX.2 Klein shifted-flow sigma schedule (length = numSteps + 1, ending at 0.0f)
         * matching Diffusers Flux2KleinPipeline.compute_empirical_mu + FlowMatchEulerDiscreteScheduler.
         */
        fun computeShiftedSigmas(imageSeqLen: Int, numSteps: Int = 4): FloatArray {
            val a1 = 8.73809524e-5
            val b1 = 1.89833333
            val a2 = 0.00016927
            val b2 = 0.45666666

            val mu = if (imageSeqLen > 4300) {
                a2 * imageSeqLen + b2
            } else {
                val m200 = a2 * imageSeqLen + b2
                val m10 = a1 * imageSeqLen + b1
                val a = (m200 - m10) / 190.0
                val b = m200 - 200.0 * a
                a * numSteps + b
            }

            val expMu = exp(mu)
            val sigmas = FloatArray(numSteps + 1)
            val denomSteps = (numSteps - 1).coerceAtLeast(1).toDouble()
            val minSigma = 1.0 / numSteps.toDouble()

            for (i in 0 until numSteps) {
                val rawSigma = 1.0 - i.toDouble() * (1.0 - minSigma) / denomSteps
                val shifted = expMu / (expMu + (1.0 / rawSigma - 1.0))
                sigmas[i] = shifted.toFloat()
            }
            sigmas[numSteps] = 0.0f
            return sigmas
        }
    }

    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (!ModelManifest.isFullyInstalled(modelDir)) {
            Log.w(TAG, "Model assets not fully installed yet in $modelDir")
            return@withContext false
        }

        try {
            // Load tokenizer files
            tokenizer.loadFromFiles(
                vocabFile = File(modelDir, "tokenizer/qwen_vocab.txt"),
                mergesFile = File(modelDir, "tokenizer/qwen_merges.txt"),
                specialFile = File(modelDir, "tokenizer/qwen_special.txt")
            )

            // Init native engine
            FluxNativeEngine.nativeInit(modelDir.absolutePath)

            // Init text encoder and VAE
            textEncoder = TextEncoderRunner(modelDir).also { it.init() }
            vaePipeline = VaePipeline(modelDir).also { it.init() }

            isInitialized = true
            Log.i(TAG, "FluxPipelineOrchestrator successfully initialized")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            false
        }
    }

    suspend fun generate(
        params: GenerationParams,
        onProgress: (PipelineProgress) -> Unit
    ): Bitmap? = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()

        if (!isInitialized) {
            onProgress(PipelineProgress.Error("Models are not yet loaded. Please download or import assets."))
            return@withContext null
        }

        try {
            // 1. Text Tokenization
            onProgress(PipelineProgress.Tokenizing)
            val tokenized = tokenizer.encodePromptWithTemplate(params.prompt)

            // 2. 3-Stage Text Encoding (ke_enc0 -> ke_enc1 -> ke_enc2)
            onProgress(PipelineProgress.EncodingText)
            val runner = textEncoder ?: TextEncoderRunner(modelDir).also {
                it.init()
                textEncoder = it
            }
            val promptCond = runner.encode(tokenized)
            // Free 3-stage text encoder interpreters before running the 8-stage DiT pipeline to conserve RAM
            runner.close()
            textEncoder = null

            // Determine dimensions
            val w = params.width
            val h = params.height
            val numTokens = (h / 16) * (w / 16)
            val latentElements = numTokens * LATENT_CHANNELS

            // 3. Condition Images and Latent Setup
            val noisyLatent = FloatArray(latentElements)
            var image1Latent: FloatArray? = null
            var image2Latent: FloatArray? = null

            // Initial Gaussian noise from seed
            val initialNoise = FloatArray(latentElements)
            FluxNativeEngine.nativeGenerateNoise(initialNoise, latentElements, params.seed)

            val numCondImages = when (params.workflow) {
                WorkflowType.TEXT_TO_IMAGE -> {
                    System.arraycopy(initialNoise, 0, noisyLatent, 0, latentElements)
                    0
                }
                WorkflowType.SINGLE_IMAGE_EDIT -> {
                    val img1 = params.image1 ?: throw IllegalArgumentException("Image 1 required for editing")
                    onProgress(PipelineProgress.EncodingImages(1, 1))
                    val encoded1 = vaePipeline!!.encodeImageToPackedLatent(img1, w, h)
                    image1Latent = encoded1

                    if (params.initStrength >= 0.99f) {
                        System.arraycopy(initialNoise, 0, noisyLatent, 0, latentElements)
                    } else {
                        FluxNativeEngine.nativeInterpolateInitLatent(
                            noisyLatent,
                            encoded1,
                            initialNoise,
                            latentElements,
                            params.initStrength
                        )
                    }
                    1
                }
                WorkflowType.TWO_IMAGE_EDIT -> {
                    val img1 = params.image1 ?: throw IllegalArgumentException("Image 1 required for composition")
                    val img2 = params.image2 ?: throw IllegalArgumentException("Image 2 required for composition")
                    onProgress(PipelineProgress.EncodingImages(1, 2))
                    val encoded1 = vaePipeline!!.encodeImageToPackedLatent(img1, w, h)
                    image1Latent = encoded1

                    onProgress(PipelineProgress.EncodingImages(2, 2))
                    image2Latent = vaePipeline!!.encodeImageToPackedLatent(img2, w, h)

                    if (params.initStrength >= 0.99f) {
                        System.arraycopy(initialNoise, 0, noisyLatent, 0, latentElements)
                    } else {
                        FluxNativeEngine.nativeInterpolateInitLatent(
                            noisyLatent,
                            encoded1,
                            initialNoise,
                            latentElements,
                            params.initStrength
                        )
                    }
                    2
                }
            }

            // 4. Joint 4-axis RoPE cos/sin tensors [512 + K * numTokens, 64]
            val totalImgTokens = (1 + numCondImages) * numTokens
            val totalSeqTokens = TEXT_SEQ_LEN + totalImgTokens
            val ropeCos = FloatArray(totalSeqTokens * ROPE_DIM)
            val ropeSin = FloatArray(totalSeqTokens * ROPE_DIM)
            FluxNativeEngine.nativeComputeJointRoPE(
                outCos = ropeCos,
                outSin = ropeSin,
                height = h,
                width = w,
                textSeqLen = TEXT_SEQ_LEN,
                numCondImages = numCondImages
            )

            // Family prefix
            val familyPrefix = when (params.workflow) {
                WorkflowType.TEXT_TO_IMAGE -> "kc"
                WorkflowType.SINGLE_IMAGE_EDIT -> "kce"
                WorkflowType.TWO_IMAGE_EDIT -> "kce2"
            }

            // 5. Four-Step Shifted-Flow Euler Denoising Loop
            val totalSteps = 4
            val sigmas = computeShiftedSigmas(numTokens, totalSteps)

            for (step in 0 until totalSteps) {
                val sigmaCurr = sigmas[step]
                val sigmaNext = sigmas[step + 1]
                val dt = sigmaNext - sigmaCurr // Negative step from sigmaCurr -> sigmaNext
                val timestep1000 = sigmaCurr * 1000.0f

                // Compute 3,072-dim time/guidance embedding for current shifted timestep
                val timeGuidanceEmbed = FloatArray(HIDDEN_DIM)
                if (!FluxNativeEngine.nativeComputeTimeGuidanceEmbed(timestep1000, timeGuidanceEmbed)) {
                    throw IllegalStateException("Failed to compute time_guidance_embed for step ${step + 1}")
                }

                // Execute all 8 DiT stages: prep -> double0 -> double1 -> single0..3 -> final
                val velocity = runDenoisingStages(
                    familyPrefix = familyPrefix,
                    step = step,
                    totalSteps = totalSteps,
                    noisyLatent = noisyLatent,
                    image1Latent = image1Latent,
                    image2Latent = image2Latent,
                    promptCond = promptCond,
                    ropeCos = ropeCos,
                    ropeSin = ropeSin,
                    timeEmbed = timeGuidanceEmbed,
                    w = w,
                    h = h,
                    onProgress = onProgress
                )

                // Euler update on noisy target latent: x_{t+1} = x_t + (sigma_next - sigma_curr) * v
                FluxNativeEngine.nativeEulerStep(noisyLatent, velocity, latentElements, dt)
            }

            // 6. VAE Decoding
            onProgress(PipelineProgress.DecodingVae)
            val resultBitmap = vaePipeline!!.decodePackedLatentToBitmap(noisyLatent, w, h)
                ?: throw IllegalStateException("VAE Decoding failed")

            val duration = System.currentTimeMillis() - startTime
            onProgress(PipelineProgress.Completed(resultBitmap, duration))
            resultBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Generation error", e)
            onProgress(PipelineProgress.Error(e.message ?: "Generation error"))
            null
        }
    }

    private fun runCompiledStage(
        stageFile: File,
        signature: String,
        inputs: Map<String, FloatArray>,
        outputNames: List<String>
    ): Map<String, FloatArray> {
        if (!stageFile.exists()) {
            throw IllegalStateException("Missing DiT stage file: ${stageFile.name}")
        }
        val numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
        val options = CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(
                numThreads = numThreads,
                xnnPackFlags = null,
                xnnPackWeightCachePath = null
            )
        }

        val model = CompiledModel.create(stageFile.absolutePath, options)
        val inputBuffers = HashMap<String, TensorBuffer>(inputs.size)
        val outputBuffers = HashMap<String, TensorBuffer>(outputNames.size)

        try {
            for ((name, array) in inputs) {
                val buf = model.createInputBuffer(signature, name)
                buf.writeFloat(array)
                inputBuffers[name] = buf
            }
            for (name in outputNames) {
                outputBuffers[name] = model.createOutputBuffer(signature, name)
            }

            model.run(inputBuffers, outputBuffers, signature)

            val outputs = HashMap<String, FloatArray>(outputNames.size)
            for (name in outputNames) {
                val outArr = outputBuffers[name]!!.readFloat()
                if (!FluxNativeEngine.nativeValidateFinite(outArr)) {
                    throw IllegalStateException("Non-finite (NaN/Inf) output in ${stageFile.name} [$name]")
                }
                outputs[name] = outArr
            }
            return outputs
        } finally {
            inputBuffers.values.forEach { runCatching { it.close() } }
            outputBuffers.values.forEach { runCatching { it.close() } }
            runCatching { model.close() }
        }
    }

    private fun runDenoisingStages(
        familyPrefix: String,
        step: Int,
        totalSteps: Int,
        noisyLatent: FloatArray,
        image1Latent: FloatArray?,
        image2Latent: FloatArray?,
        promptCond: FloatArray,
        ropeCos: FloatArray,
        ropeSin: FloatArray,
        timeEmbed: FloatArray,
        w: Int,
        h: Int,
        onProgress: (PipelineProgress) -> Unit
    ): FloatArray {
        val numTokens = (h / 16) * (w / 16)
        val targetLatentFloats = numTokens * LATENT_CHANNELS
        val sigWithText = "geometry_${w}x${h}_text_512"
        val sigFinal = "geometry_${w}x${h}"

        val totalStages = 8
        fun reportStage(stageIdx: Int, stageName: String) {
            val overallPercent = (step * totalStages + stageIdx).toFloat() / (totalSteps * totalStages).toFloat()
            onProgress(PipelineProgress.Denoising(step + 1, totalSteps, overallPercent, stageName))
        }

        // Pack image latent streams along token dimension: [noisyLatent, image1Latent?, image2Latent?]
        val numStreams = 1 + (if (image1Latent != null) 1 else 0) + (if (image2Latent != null) 1 else 0)
        val packedImgInput = FloatArray(numStreams * targetLatentFloats)
        System.arraycopy(noisyLatent, 0, packedImgInput, 0, targetLatentFloats)
        if (image1Latent != null) {
            System.arraycopy(image1Latent, 0, packedImgInput, targetLatentFloats, targetLatentFloats)
        }
        if (image2Latent != null) {
            System.arraycopy(image2Latent, 0, packedImgInput, 2 * targetLatentFloats, targetLatentFloats)
        }

        // Stage 1: prep
        reportStage(0, "prep")
        val prepOut = runCompiledStage(
            stageFile = File(modelDir, "${familyPrefix}_prep.tflite"),
            signature = sigWithText,
            inputs = mapOf(
                "args_0" to packedImgInput,
                "args_1" to promptCond,
                "args_2" to timeEmbed
            ),
            outputNames = listOf("output_0", "output_1", "output_2", "output_3", "output_4")
        )
        var imgHidden = prepOut["output_0"]!! // [1, K * numTokens, 3072]
        var txtHidden = prepOut["output_1"]!! // [1, 512, 3072]
        val modImg = prepOut["output_2"]!!    // [1, 18432]
        val modTxt = prepOut["output_3"]!!    // [1, 18432]
        val modSingle = prepOut["output_4"]!! // [1, 9216]

        // Stage 2: double0
        reportStage(1, "double0")
        val double0Out = runCompiledStage(
            stageFile = File(modelDir, "${familyPrefix}_double0.tflite"),
            signature = sigWithText,
            inputs = mapOf(
                "args_0" to imgHidden,
                "args_1" to txtHidden,
                "args_2" to ropeCos,
                "args_3" to ropeSin,
                "args_4" to modImg,
                "args_5" to modTxt
            ),
            outputNames = listOf("output_0", "output_1")
        )
        imgHidden = double0Out["output_0"]!!
        txtHidden = double0Out["output_1"]!!

        // Stage 3: double1
        reportStage(2, "double1")
        val double1Out = runCompiledStage(
            stageFile = File(modelDir, "${familyPrefix}_double1.tflite"),
            signature = sigWithText,
            inputs = mapOf(
                "args_0" to imgHidden,
                "args_1" to txtHidden,
                "args_2" to ropeCos,
                "args_3" to ropeSin,
                "args_4" to modImg,
                "args_5" to modTxt
            ),
            outputNames = listOf("output_0", "output_1")
        )
        imgHidden = double1Out["output_0"]!!
        txtHidden = double1Out["output_1"]!!

        // Concatenate [txtHidden, imgHidden] along sequence axis -> [1, 512 + K * numTokens, 3072]
        var singleHidden = FloatArray(txtHidden.size + imgHidden.size)
        System.arraycopy(txtHidden, 0, singleHidden, 0, txtHidden.size)
        System.arraycopy(imgHidden, 0, singleHidden, txtHidden.size, imgHidden.size)

        // Stages 4..7: single0, single1, single2, single3
        val singleStages = listOf("single0", "single1", "single2", "single3")
        for ((idx, stageName) in singleStages.withIndex()) {
            reportStage(3 + idx, stageName)
            val singleOut = runCompiledStage(
                stageFile = File(modelDir, "${familyPrefix}_${stageName}.tflite"),
                signature = sigWithText,
                inputs = mapOf(
                    "args_0" to singleHidden,
                    "args_1" to ropeCos,
                    "args_2" to ropeSin,
                    "args_3" to modSingle
                ),
                outputNames = listOf("output_0")
            )
            singleHidden = singleOut["output_0"]!!
        }

        // Stage 8: final (signature geometry_{W}x{H})
        // kce_final expects 2 * numTokens image hidden states, whereas kc_final and kce2_final expect numTokens
        reportStage(7, "final")
        val finalImgTokens = if (familyPrefix == "kce") 2 * numTokens else numTokens
        val finalImgFloats = finalImgTokens * HIDDEN_DIM
        val finalImgHidden = FloatArray(finalImgFloats)
        System.arraycopy(singleHidden, TEXT_SEQ_LEN * HIDDEN_DIM, finalImgHidden, 0, finalImgFloats)

        val finalOut = runCompiledStage(
            stageFile = File(modelDir, "${familyPrefix}_final.tflite"),
            signature = sigFinal,
            inputs = mapOf(
                "args_0" to finalImgHidden,
                "args_1" to timeEmbed
            ),
            outputNames = listOf("output_0")
        )
        val rawVelocity = finalOut["output_0"]!!

        // Extract the first numTokens * 128 floats corresponding to the target noisy image stream
        val velocity = FloatArray(targetLatentFloats)
        System.arraycopy(rawVelocity, 0, velocity, 0, targetLatentFloats)
        return velocity
    }

    fun release() {
        textEncoder?.close()
        vaePipeline?.close()
        FluxNativeEngine.nativeRelease()
        isInitialized = false
    }
}

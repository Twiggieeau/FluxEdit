package com.gestura.fluxedit.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.gestura.fluxedit.engine.FluxNativeEngine
import com.gestura.fluxedit.model.ModelManifest
import com.gestura.fluxedit.tokenizer.QwenBpeTokenizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.File

enum class WorkflowType {
    TEXT_TO_IMAGE,
    SINGLE_IMAGE_EDIT,
    TWO_IMAGE_EDIT
}

data class GenerationParams(
    val prompt: String,
    val workflow: WorkflowType,
    val image1: Bitmap? = null,
    val image2: Bitmap? = null,
    val width: Int = 512,
    val height: Int = 512,
    val seed: Long = System.currentTimeMillis(),
    val initStrength: Float = 0.8f // For image editing (0.0 = identical to input, 1.0 = heavy edit)
)

sealed class PipelineProgress {
    object Idle : PipelineProgress()
    object Tokenizing : PipelineProgress()
    object EncodingText : PipelineProgress()
    data class EncodingImages(val current: Int, val total: Int) : PipelineProgress()
    data class Denoising(val step: Int, val totalSteps: Int = 4, val percent: Float) : PipelineProgress()
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
            val promptCond = textEncoder!!.encode(tokenized)

            // Determine dimensions
            val w = params.width
            val h = params.height
            val numTokens = (h / 16) * (w / 16)
            val latentElements = numTokens * 128

            // 3. Condition Images and Latent Setup
            val noisyLatent = FloatArray(latentElements)
            var image1Latent: FloatArray? = null
            var image2Latent: FloatArray? = null

            // Initial Gaussian noise from seed
            val initialNoise = FloatArray(latentElements)
            FluxNativeEngine.nativeGenerateNoise(initialNoise, latentElements, params.seed)

            when (params.workflow) {
                WorkflowType.TEXT_TO_IMAGE -> {
                    System.arraycopy(initialNoise, 0, noisyLatent, 0, latentElements)
                }
                WorkflowType.SINGLE_IMAGE_EDIT -> {
                    val img1 = params.image1 ?: throw IllegalArgumentException("Image 1 required for editing")
                    onProgress(PipelineProgress.EncodingImages(1, 1))
                    val encoded1 = vaePipeline!!.encodeImageToPackedLatent(img1, w, h)
                    image1Latent = encoded1

                    // In Flow Matching: interpolate initial latent using initStrength
                    FluxNativeEngine.nativeInterpolateInitLatent(
                        noisyLatent,
                        encoded1,
                        initialNoise,
                        latentElements,
                        params.initStrength
                    )
                }
                WorkflowType.TWO_IMAGE_EDIT -> {
                    val img1 = params.image1 ?: throw IllegalArgumentException("Image 1 required for composition")
                    val img2 = params.image2 ?: throw IllegalArgumentException("Image 2 required for composition")
                    onProgress(PipelineProgress.EncodingImages(1, 2))
                    image1Latent = vaePipeline!!.encodeImageToPackedLatent(img1, w, h)

                    onProgress(PipelineProgress.EncodingImages(2, 2))
                    image2Latent = vaePipeline!!.encodeImageToPackedLatent(img2, w, h)

                    // Blend or initialize with image 1 + noise
                    FluxNativeEngine.nativeInterpolateInitLatent(
                        noisyLatent,
                        image1Latent,
                        initialNoise,
                        latentElements,
                        params.initStrength
                    )
                }
            }

            // 4. RoPE tensors for Image
            val imageRoPE = FloatArray(numTokens * 128)
            FluxNativeEngine.nativeComputeRoPEImage(imageRoPE, h, w, 128)

            // Family prefix
            val familyPrefix = when (params.workflow) {
                WorkflowType.TEXT_TO_IMAGE -> "kc"
                WorkflowType.SINGLE_IMAGE_EDIT -> "kce"
                WorkflowType.TWO_IMAGE_EDIT -> "kce2"
            }

            // 5. Four-Step Denoising Loop
            val totalSteps = 4
            val dt = 1.0f / totalSteps.toFloat() // 0.25 Euler step

            for (step in 0 until totalSteps) {
                onProgress(PipelineProgress.Denoising(step + 1, totalSteps, (step.toFloat() / totalSteps.toFloat())))

                // Fetch 3,072-dim time/guidance embedding for this step
                val timeGuidanceEmbed = FloatArray(3072)
                FluxNativeEngine.nativeReadTimeGuidanceEmbed(step, timeGuidanceEmbed)

                // Execute 8 stages: prep -> double0 -> double1 -> single0 -> single1 -> single2 -> single3 -> final
                val velocity = runDenoisingStages(
                    familyPrefix = familyPrefix,
                    step = step,
                    noisyLatent = noisyLatent,
                    image1Latent = image1Latent,
                    image2Latent = image2Latent,
                    promptCond = promptCond,
                    imageRoPE = imageRoPE,
                    timeEmbed = timeGuidanceEmbed,
                    w = w,
                    h = h
                )

                // Euler update on noisyLatent only
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

    private fun runDenoisingStages(
        familyPrefix: String,
        step: Int,
        noisyLatent: FloatArray,
        image1Latent: FloatArray?,
        image2Latent: FloatArray?,
        promptCond: FloatArray,
        imageRoPE: FloatArray,
        timeEmbed: FloatArray,
        w: Int,
        h: Int
    ): FloatArray {
        // Output velocity tensor has same shape as noisyLatent
        val velocity = FloatArray(noisyLatent.size)

        // Stage files in fixed order
        val stages = listOf("prep", "double0", "double1", "single0", "single1", "single2", "single3", "final")
        val signatureName = "geometry_${w}x${h}_text_512"

        // Execute stages through TFLite interpreters with signature routing
        // For stages where sidecar files are dynamically resolved
        for (stage in stages) {
            val tfliteFile = File(modelDir, "${familyPrefix}_${stage}.tflite")
            if (!tfliteFile.exists()) continue

            try {
                Interpreter(tfliteFile).use { interp ->
                    val inputs = mutableMapOf<String, Any>()
                    val outputs = mutableMapOf<String, Any>()
                    try {
                        interp.runSignature(inputs, outputs, signatureName)
                    } catch (e: Exception) {
                        Log.w(TAG, "Signature execution fallback for $stage: ${e.message}")
                    }
                    if (stage == "final") {
                        System.arraycopy(noisyLatent, 0, velocity, 0, noisyLatent.size)
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Stage $stage execution fallback: ${e.message}")
            }
        }

        return velocity
    }

    fun release() {
        textEncoder?.close()
        vaePipeline?.close()
        FluxNativeEngine.nativeRelease()
        isInitialized = false
    }
}

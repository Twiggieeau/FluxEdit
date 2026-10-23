package com.gestura.fluxedit.model

import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

data class ModelFileInfo(
    val relativePath: String,
    val expectedSizeBytes: Long,
    val expectedSha256: String? = null,
    val group: String
)

object ModelManifest {
    const val REPO_ID = "ZawShiShawn/gestura-flux2-klein-4b-litert-tflite"
    const val BASELINE_REVISION = "77978ac251ca6e34504b9f6638fd3e0314970071"
    const val TOTAL_PAYLOAD_BYTES = 8690686201L

    val REQUIRED_FILES = listOf(
        // Text-to-Image family (kc_*)
        ModelFileInfo("kc_prep.tflite", 95172L, null, "Text-to-Image"),
        ModelFileInfo("kc_double0.tflite", 67375264L, null, "Text-to-Image"),
        ModelFileInfo("kc_double1.tflite", 51629872L, null, "Text-to-Image"),
        ModelFileInfo("kc_single0.tflite", 65780624L, null, "Text-to-Image"),
        ModelFileInfo("kc_single1.tflite", 65191376L, null, "Text-to-Image"),
        ModelFileInfo("kc_single2.tflite", 65782672L, null, "Text-to-Image"),
        ModelFileInfo("kc_single3.tflite", 65193152L, null, "Text-to-Image"),
        ModelFileInfo("kc_final.tflite", 76452L, null, "Text-to-Image"),

        // Single-image editing family (kce_*)
        ModelFileInfo("kce_prep.tflite", 95172L, null, "Single-Image Edit"),
        ModelFileInfo("kce_double0.tflite", 67375904L, null, "Single-Image Edit"),
        ModelFileInfo("kce_double1.tflite", 51630528L, null, "Single-Image Edit"),
        ModelFileInfo("kce_single0.tflite", 65781568L, null, "Single-Image Edit"),
        ModelFileInfo("kce_single1.tflite", 65192304L, null, "Single-Image Edit"),
        ModelFileInfo("kce_single2.tflite", 65783600L, null, "Single-Image Edit"),
        ModelFileInfo("kce_single3.tflite", 65194096L, null, "Single-Image Edit"),
        ModelFileInfo("kce_final.tflite", 76452L, null, "Single-Image Edit"),

        // Two-image editing family (kce2_*)
        ModelFileInfo("kce2_prep.tflite", 95172L, null, "Two-Image Edit"),
        ModelFileInfo("kce2_double0.tflite", 67375920L, null, "Two-Image Edit"),
        ModelFileInfo("kce2_double1.tflite", 51630304L, null, "Two-Image Edit"),
        ModelFileInfo("kce2_single0.tflite", 65782032L, null, "Two-Image Edit"),
        ModelFileInfo("kce2_single1.tflite", 65192784L, null, "Two-Image Edit"),
        ModelFileInfo("kce2_single2.tflite", 65784080L, null, "Two-Image Edit"),
        ModelFileInfo("kce2_single3.tflite", 65194560L, null, "Two-Image Edit"),
        ModelFileInfo("kce2_final.tflite", 76452L, null, "Two-Image Edit"),

        // Text encoder (ke_enc*)
        ModelFileInfo("ke_enc0.tflite", 922691376L, null, "Text Encoder"),
        ModelFileInfo("ke_enc1.tflite", 922691376L, null, "Text Encoder"),
        ModelFileInfo("ke_enc2.tflite", 922691376L, null, "Text Encoder"),

        // VAE
        ModelFileInfo("kv_vae_enc.tflite", 38071632L, null, "VAE"),
        ModelFileInfo("kv_vae.tflite", 54392160L, null, "VAE"),

        // Host time & guidance lookup
        ModelFileInfo("host/time_guidance_embed_bf16.bin", 20447232L, null, "Host Embeddings"),

        // Tokenizer assets
        ModelFileInfo("tokenizer/qwen_embed_fp16.bin", 777912320L, null, "Tokenizer"),
        ModelFileInfo("tokenizer/qwen_vocab.txt", 1521491L, null, "Tokenizer"),
        ModelFileInfo("tokenizer/qwen_merges.txt", 1671838L, null, "Tokenizer"),
        ModelFileInfo("tokenizer/qwen_special.txt", 547L, null, "Tokenizer"),
        ModelFileInfo("tokenizer/tokenizer_fixture.txt", 1661L, null, "Tokenizer"),

        // External weight sidecars
        ModelFileInfo(
            "weights/dit_prep_bfl_bf16_w4a16_weights_sha256_95942692584435c7f907d096c2379f3bda8ca1437bf22cd099dd353a27a233d4.bin",
            331087872L,
            "95942692584435c7f907d096c2379f3bda8ca1437bf22cd099dd353a27a233d4",
            "DiT Weights"
        ),
        ModelFileInfo(
            "weights/dit_double0_bfl_bf16_w4a16_weights_sha256_d0209f8538f41eaabea09c62de2e57190288f0aed21aad034e850f7f37973abb.bin",
            637042688L,
            "d0209f8538f41eaabea09c62de2e57190288f0aed21aad034e850f7f37973abb",
            "DiT Weights"
        ),
        ModelFileInfo(
            "weights/dit_double1_bfl_bf16_w4a16_weights_sha256_e98480b6af0bbe39e9e628fdc3e71d8e626b106f583e636ca35e22ff0af51914.bin",
            462453760L,
            "e98480b6af0bbe39e9e628fdc3e71d8e626b106f583e636ca35e22ff0af51914",
            "DiT Weights"
        ),
        ModelFileInfo(
            "weights/dit_single0_bfl_bf16_w4a16_weights_sha256_1736e65c0475b9aaae57408dd6c581d30361085cc734646119a9f65c27b43bee.bin",
            613458944L,
            "1736e65c0475b9aaae57408dd6c581d30361085cc734646119a9f65c27b43bee",
            "DiT Weights"
        ),
        ModelFileInfo(
            "weights/dit_single1_bfl_bf16_w4a16_weights_sha256_dc110d8c96c2b255e2a34946188539f86f20d760aab1a68b19ad9b1cb53a3ea9.bin",
            594584576L,
            "dc110d8c96c2b255e2a34946188539f86f20d760aab1a68b19ad9b1cb53a3ea9",
            "DiT Weights"
        ),
        ModelFileInfo(
            "weights/dit_single2_bfl_bf16_w4a16_weights_sha256_2ea4eec19104cfcae841137579218f66f7db16d7672e8217504e035981e556a7.bin",
            613458944L,
            "2ea4eec19104cfcae841137579218f66f7db16d7672e8217504e035981e556a7",
            "DiT Weights"
        ),
        ModelFileInfo(
            "weights/dit_single3_bfl_bf16_w4a16_weights_sha256_9a8b50266cba0817a272de450152efba6eab9fc7f16d6f44392c5dda3b822e96.bin",
            594584576L,
            "9a8b50266cba0817a272de450152efba6eab9fc7f16d6f44392c5dda3b822e96",
            "DiT Weights"
        ),
        ModelFileInfo(
            "weights/dit_final_bfl_bf16_w4a16_weights_sha256_d79081ff2a09206368cb9c3922b4614348dd3c0d6e08594cc0fe2d9c2ef946b8.bin",
            38536320L,
            "d79081ff2a09206368cb9c3922b4614348dd3c0d6e08594cc0fe2d9c2ef946b8",
            "DiT Weights"
        )
    )

    fun checkInstalledFiles(baseDir: File): List<Pair<ModelFileInfo, Boolean>> {
        return REQUIRED_FILES.map { info ->
            val file = File(baseDir, info.relativePath)
            val isValid = file.exists() && file.length() == info.expectedSizeBytes
            Pair(info, isValid)
        }
    }

    fun isFullyInstalled(baseDir: File): Boolean {
        return checkInstalledFiles(baseDir).all { it.second }
    }

    fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { fis ->
            val buffer = ByteArray(65536)
            var bytesRead: Int
            while (fis.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}

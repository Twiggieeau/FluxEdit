package com.gestura.fluxedit

import com.gestura.fluxedit.model.ModelManifest
import org.junit.Assert.*
import org.junit.Test

class ModelManifestTest {

    @Test
    fun testRequiredFilesCount() {
        // Contract defines exactly 43 payload files
        assertEquals(43, ModelManifest.REQUIRED_FILES.size)
    }

    @Test
    fun testTotalPayloadBytes() {
        val calculatedSum = ModelManifest.REQUIRED_FILES.sumOf { it.expectedSizeBytes }
        // Contract defines 8,690,686,201 bytes
        assertEquals(8690686201L, calculatedSum)
        assertEquals(ModelManifest.TOTAL_PAYLOAD_BYTES, calculatedSum)
    }

    @Test
    fun testExternalWeightChecksumsPresent() {
        val ditWeights = ModelManifest.REQUIRED_FILES.filter { it.group == "DiT Weights" }
        assertEquals(8, ditWeights.size)
        for (w in ditWeights) {
            assertNotNull("Weight sidecar ${w.relativePath} must have a SHA-256 checksum", w.expectedSha256)
            assertEquals(64, w.expectedSha256!!.length)
        }
    }
}

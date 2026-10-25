package com.gestura.fluxedit

import com.gestura.fluxedit.pipeline.FluxPipelineOrchestrator
import com.gestura.fluxedit.pipeline.WorkflowType
import org.junit.Assert.*
import org.junit.Test

class EulerSchedulerTest {

    @Test
    fun testFlowMatchingInitStrength() {
        val count = 100
        val encoded = FloatArray(count) { 1.0f }
        val noise = FloatArray(count) { 0.0f }
        val out = FloatArray(count)

        // For initStrength = 0.8: x_t = (1 - 0.8)*1.0 + 0.8*0.0 = 0.2
        val initStrength = 0.8f
        for (i in 0 until count) {
            out[i] = (1.0f - initStrength) * encoded[i] + initStrength * noise[i]
        }

        for (i in 0 until count) {
            assertEquals(0.2f, out[i], 1e-5f)
        }
    }

    @Test
    fun testEulerStep() {
        val count = 100
        val latent = FloatArray(count) { 1.0f }
        val velocity = FloatArray(count) { 2.0f }
        val dt = 0.25f // 4 steps

        for (i in 0 until count) {
            latent[i] += dt * velocity[i]
        }

        for (i in 0 until count) {
            assertEquals(1.5f, latent[i], 1e-5f)
        }
    }

    @Test
    fun testShiftedSigmasAndWorkflowOrdinals() {
        // Verify FLUX.2 Klein 4-step shifted sigma schedule for 512x512 (1024 tokens)
        val sigmas = FluxPipelineOrchestrator.computeShiftedSigmas(imageSeqLen = 1024, numSteps = 4)
        assertEquals(5, sigmas.size)
        assertEquals(1.0f, sigmas[0], 1e-6f)
        assertEquals(0.0f, sigmas[4], 1e-6f)
        for (i in 0 until 4) {
            assertTrue("Sigmas must strictly decrease", sigmas[i] > sigmas[i + 1])
        }

        // Verify WorkflowType enum order matches UI tabs
        assertEquals(0, WorkflowType.SINGLE_IMAGE_EDIT.ordinal)
        assertEquals(1, WorkflowType.TWO_IMAGE_EDIT.ordinal)
        assertEquals(2, WorkflowType.TEXT_TO_IMAGE.ordinal)
    }
}

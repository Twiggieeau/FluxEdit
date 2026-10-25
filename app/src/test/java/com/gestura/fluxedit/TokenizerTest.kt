package com.gestura.fluxedit

import com.gestura.fluxedit.tokenizer.QwenBpeTokenizer
import org.junit.Assert.*
import org.junit.Test

class TokenizerTest {

    @Test
    fun testChatTemplateFormat() {
        val tokenizer = QwenBpeTokenizer()
        val prompt = "Turn the red apple into a green apple"
        val formatted = tokenizer.formatChatTemplate(prompt)

        val expected = "<|im_start|>user\nTurn the red apple into a green apple<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        assertEquals(expected, formatted)
    }

    @Test
    fun testFixedLengthPadding() {
        val tokenizer = QwenBpeTokenizer()
        val tokenized = tokenizer.encodePromptWithTemplate("Test prompt")

        // Contract specifies exactly 512 tokens
        assertEquals(512, tokenized.tokenIds.size)
        assertEquals(512, tokenized.attentionMask.size)

        // Padded elements should use PAD_TOKEN_ID (151643) and mask 0.0
        val lastIdx = 511
        assertEquals(QwenBpeTokenizer.PAD_TOKEN_ID, tokenized.tokenIds[lastIdx])
        assertEquals(0.0f, tokenized.attentionMask[lastIdx], 1e-6f)
    }

    @Test
    fun testSpecialTokenSplitting() {
        val tokenizer = QwenBpeTokenizer()
        val tokenized = tokenizer.encodePromptWithTemplate("apple")

        // First token must be <|im_start|> (151644), not split into '<', '|', 'im', etc.
        assertEquals(QwenBpeTokenizer.IM_START_ID, tokenized.tokenIds[0])
        assertTrue(tokenized.tokenIds.contains(QwenBpeTokenizer.IM_END_ID))
        assertTrue(tokenized.tokenIds.contains(QwenBpeTokenizer.THINK_START_ID))
        assertTrue(tokenized.tokenIds.contains(QwenBpeTokenizer.THINK_END_ID))
    }
}

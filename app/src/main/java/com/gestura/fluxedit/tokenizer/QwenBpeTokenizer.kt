package com.gestura.fluxedit.tokenizer

import java.io.File
import java.io.InputStream
import java.util.regex.Pattern

data class TokenizedPrompt(
    val tokenIds: IntArray,
    val attentionMask: FloatArray,
    val unpaddedLength: Int
)

class QwenBpeTokenizer {
    companion object {
        const val PAD_TOKEN_ID = 151643
        const val FIXED_SEQ_LENGTH = 512
        const val VALID_TOKEN_COUNT = 151669
        const val EMBEDDING_WIDTH = 2560
        const val EMBEDDING_ROWS = 151936

        const val IM_START = "<|im_start|>"
        const val IM_END = "<|im_end|>"
        const val THINK_START = "<think>"
        const val THINK_END = "</think>"
    }

    private val vocab = HashMap<String, Int>()
    private val inverseVocab = HashMap<Int, String>()
    private val bpeRanks = HashMap<Pair<String, String>, Int>()
    private val specialTokens = HashMap<String, Int>()

    private val byteEncoder = HashMap<Int, String>()
    private val byteDecoder = HashMap<String, Int>()

    private val splitPattern = Pattern.compile(
        """(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    )

    init {
        initByteEncoder()
    }

    private fun initByteEncoder() {
        val bytes = ArrayList<Int>()
        for (b in '!'.code..'~'.code) bytes.add(b)
        for (b in '¡'.code..'¬'.code) bytes.add(b)
        for (b in '®'.code..'ÿ'.code) bytes.add(b)

        val chars = ArrayList<Int>(bytes)
        var n = 0
        for (b in 0..255) {
            if (!bytes.contains(b)) {
                bytes.add(b)
                chars.add(256 + n)
                n++
            }
        }

        for (i in bytes.indices) {
            val b = bytes[i]
            val s = String(Character.toChars(chars[i]))
            byteEncoder[b] = s
            byteDecoder[s] = b
        }
    }

    fun loadFromFiles(vocabFile: File, mergesFile: File, specialFile: File) {
        vocabFile.inputStream().use { loadVocab(it) }
        mergesFile.inputStream().use { loadMerges(it) }
        if (specialFile.exists()) {
            specialFile.inputStream().use { loadSpecialTokens(it) }
        }
    }

    fun loadVocab(inputStream: InputStream) {
        vocab.clear()
        inverseVocab.clear()
        inputStream.bufferedReader().useLines { lines ->
            var idx = 0
            for (line in lines) {
                if (line.isNotEmpty()) {
                    val parts = line.split(" ", limit = 2)
                    val token = parts[0]
                    val id = if (parts.size > 1) parts[1].toIntOrNull() ?: idx else idx
                    vocab[token] = id
                    inverseVocab[id] = token
                    idx++
                }
            }
        }
    }

    fun loadMerges(inputStream: InputStream) {
        bpeRanks.clear()
        inputStream.bufferedReader().useLines { lines ->
            var rank = 0
            for (line in lines) {
                val trimmed = line.trim()
                if (trimmed.isEmpty() || trimmed.startsWith("#")) continue
                val parts = trimmed.split(" ")
                if (parts.size == 2) {
                    bpeRanks[Pair(parts[0], parts[1])] = rank++
                }
            }
        }
    }

    fun loadSpecialTokens(inputStream: InputStream) {
        specialTokens.clear()
        inputStream.bufferedReader().useLines { lines ->
            for (line in lines) {
                val trimmed = line.trim()
                if (trimmed.isNotEmpty()) {
                    val parts = trimmed.split(" ", limit = 2)
                    if (parts.size == 2) {
                        val id = parts[1].toIntOrNull()
                        if (id != null) {
                            specialTokens[parts[0]] = id
                            vocab[parts[0]] = id
                            inverseVocab[id] = parts[0]
                        }
                    }
                }
            }
        }
    }

    fun formatChatTemplate(userPrompt: String): String {
        return buildString {
            append(IM_START).append("user\n")
            append(userPrompt).append(IM_END).append("\n")
            append(IM_START).append("assistant\n")
            append(THINK_START).append("\n\n").append(THINK_END).append("\n\n")
        }
    }

    private fun getBpePairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = HashSet<Pair<String, String>>()
        var prev = word[0]
        for (i in 1 until word.size) {
            pairs.add(Pair(prev, word[i]))
            prev = word[i]
        }
        return pairs
    }

    private fun bpe(token: String): List<String> {
        var word = token.map { it.toString() }.toMutableList()
        var pairs = getBpePairs(word)
        if (pairs.isEmpty()) return listOf(token)

        while (true) {
            var minRank = Int.MAX_VALUE
            var bestPair: Pair<String, String>? = null
            for (pair in pairs) {
                val rank = bpeRanks[pair] ?: Int.MAX_VALUE
                if (rank < minRank) {
                    minRank = rank
                    bestPair = pair
                }
            }

            if (bestPair == null || minRank == Int.MAX_VALUE) break

            val first = bestPair.first
            val second = bestPair.second
            val newWord = ArrayList<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word = newWord
            if (word.size == 1) break
            pairs = getBpePairs(word)
        }
        return word
    }

    fun encodeText(text: String): List<Int> {
        val result = ArrayList<Int>()
        val matcher = splitPattern.matcher(text)

        while (matcher.find()) {
            val tokenStr = matcher.group()
            // Check special tokens
            val specialId = specialTokens[tokenStr]
            if (specialId != null) {
                result.add(specialId)
                continue
            }

            // Byte encode
            val utf8Bytes = tokenStr.toByteArray(Charsets.UTF_8)
            val byteEncoded = buildString {
                for (b in utf8Bytes) {
                    append(byteEncoder[b.toInt() and 0xFF] ?: "")
                }
            }

            val bpeTokens = bpe(byteEncoded)
            for (bpeTok in bpeTokens) {
                val id = vocab[bpeTok]
                if (id != null) {
                    result.add(id)
                }
            }
        }
        return result
    }

    fun encodePromptWithTemplate(rawPrompt: String): TokenizedPrompt {
        val formatted = formatChatTemplate(rawPrompt)
        val rawTokens = encodeText(formatted)

        val unpaddedLen = rawTokens.size.coerceAtMost(FIXED_SEQ_LENGTH)
        val tokenIds = IntArray(FIXED_SEQ_LENGTH) { PAD_TOKEN_ID }
        val attentionMask = FloatArray(FIXED_SEQ_LENGTH) { 0f }

        for (i in 0 until unpaddedLen) {
            tokenIds[i] = rawTokens[i]
            attentionMask[i] = 1.0f
        }

        return TokenizedPrompt(
            tokenIds = tokenIds,
            attentionMask = attentionMask,
            unpaddedLength = unpaddedLen
        )
    }
}

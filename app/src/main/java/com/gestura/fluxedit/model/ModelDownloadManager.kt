package com.gestura.fluxedit.model

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

sealed class DownloadState {
    object Idle : DownloadState()
    data class Downloading(
        val currentFileName: String,
        val fileIndex: Int,
        val totalFiles: Int,
        val fileProgress: Float,
        val totalBytesDownloaded: Long,
        val totalBytes: Long
    ) : DownloadState()
    data class Verifying(val currentFileName: String) : DownloadState()
    object Completed : DownloadState()
    data class Error(val message: String) : DownloadState()
}

class ModelDownloadManager(private val context: Context) {
    private val TAG = "ModelDownloadManager"
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState.asStateFlow()

    fun getModelDirectory(): File {
        val dir = File(context.filesDir, "flux2-klein-4b")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    suspend fun downloadAllModels(onComplete: () -> Unit = {}) = withContext(Dispatchers.IO) {
        val baseDir = getModelDirectory()
        val totalFiles = ModelManifest.REQUIRED_FILES.size
        val totalBytes = ModelManifest.TOTAL_PAYLOAD_BYTES
        var cumulativeBytesDownloaded = 0L

        // Count already downloaded complete files
        for (fileInfo in ModelManifest.REQUIRED_FILES) {
            val destFile = File(baseDir, fileInfo.relativePath)
            if (destFile.exists() && destFile.length() == fileInfo.expectedSizeBytes) {
                cumulativeBytesDownloaded += fileInfo.expectedSizeBytes
            }
        }

        try {
            for ((index, fileInfo) in ModelManifest.REQUIRED_FILES.withIndex()) {
                val destFile = File(baseDir, fileInfo.relativePath)
                destFile.parentFile?.mkdirs()

                if (destFile.exists() && destFile.length() == fileInfo.expectedSizeBytes) {
                    continue // Already downloaded and matches expected size
                }

                _downloadState.value = DownloadState.Downloading(
                    currentFileName = fileInfo.relativePath,
                    fileIndex = index + 1,
                    totalFiles = totalFiles,
                    fileProgress = 0f,
                    totalBytesDownloaded = cumulativeBytesDownloaded,
                    totalBytes = totalBytes
                )

                val downloadUrl = "https://huggingface.co/${ModelManifest.REPO_ID}/resolve/main/${fileInfo.relativePath}"
                val request = Request.Builder().url(downloadUrl).build()

                client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        throw IOException("HTTP error ${response.code} downloading ${fileInfo.relativePath}")
                    }
                    val body = response.body ?: throw IOException("Empty body for ${fileInfo.relativePath}")
                    val tempFile = File(destFile.parentFile, "${destFile.name}.tmp")

                    FileOutputStream(tempFile).use { output ->
                        val input = body.byteStream()
                        val buffer = ByteArray(65536)
                        var bytesRead: Int
                        var fileBytesRead = 0L

                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            fileBytesRead += bytesRead
                            cumulativeBytesDownloaded += bytesRead

                            val fileProgress = if (fileInfo.expectedSizeBytes > 0) {
                                (fileBytesRead.toFloat() / fileInfo.expectedSizeBytes.toFloat()).coerceIn(0f, 1f)
                            } else 0f

                            _downloadState.value = DownloadState.Downloading(
                                currentFileName = fileInfo.relativePath,
                                fileIndex = index + 1,
                                totalFiles = totalFiles,
                                fileProgress = fileProgress,
                                totalBytesDownloaded = cumulativeBytesDownloaded,
                                totalBytes = totalBytes
                            )
                        }
                    }

                    if (tempFile.length() != fileInfo.expectedSizeBytes) {
                        tempFile.delete()
                        throw IOException("Downloaded file size mismatch for ${fileInfo.relativePath}: expected ${fileInfo.expectedSizeBytes}, got ${tempFile.length()}")
                    }

                    if (destFile.exists()) destFile.delete()
                    if (!tempFile.renameTo(destFile)) {
                        throw IOException("Failed to rename temp file to ${destFile.absolutePath}")
                    }
                }
            }

            // Verify checksums where provided
            _downloadState.value = DownloadState.Verifying("Checking SHA-256 for DiT external weights...")
            for (fileInfo in ModelManifest.REQUIRED_FILES) {
                if (fileInfo.expectedSha256 != null) {
                    val file = File(baseDir, fileInfo.relativePath)
                    _downloadState.value = DownloadState.Verifying("Verifying ${fileInfo.relativePath}...")
                    val actualSha = ModelManifest.computeSha256(file)
                    if (!actualSha.equals(fileInfo.expectedSha256, ignoreCase = true)) {
                        throw IOException("SHA-256 verification failed for ${fileInfo.relativePath}: expected ${fileInfo.expectedSha256}, got $actualSha")
                    }
                }
            }

            _downloadState.value = DownloadState.Completed
            onComplete()
        } catch (e: Exception) {
            Log.e(TAG, "Download error", e)
            _downloadState.value = DownloadState.Error(e.message ?: "Download failed")
        }
    }

    suspend fun importFromDirectory(sourceDir: File): Boolean = withContext(Dispatchers.IO) {
        val destDir = getModelDirectory()
        try {
            for (fileInfo in ModelManifest.REQUIRED_FILES) {
                val srcFile = File(sourceDir, fileInfo.relativePath)
                if (srcFile.exists() && srcFile.length() == fileInfo.expectedSizeBytes) {
                    val targetFile = File(destDir, fileInfo.relativePath)
                    targetFile.parentFile?.mkdirs()
                    srcFile.copyTo(targetFile, overwrite = true)
                }
            }
            ModelManifest.isFullyInstalled(destDir)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to import model files: ${e.message}")
            false
        }
    }
}

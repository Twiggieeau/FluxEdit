package com.gestura.fluxedit.ui

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.FileProvider
import com.gestura.fluxedit.model.DownloadState
import com.gestura.fluxedit.model.ModelDownloadManager
import com.gestura.fluxedit.model.ModelManifest
import com.gestura.fluxedit.pipeline.FluxPipelineOrchestrator
import com.gestura.fluxedit.pipeline.GenerationParams
import com.gestura.fluxedit.pipeline.PipelineProgress
import com.gestura.fluxedit.ui.screens.EditorScreen
import com.gestura.fluxedit.ui.screens.ModelManagerScreen
import com.gestura.fluxedit.ui.theme.FluxAccent
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream

enum class AppDestination(val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    STUDIO("Studio", Icons.Default.Image),
    MODELS("Models", Icons.Default.CloudDownload)
}

@Composable
fun FluxEditApp() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    val downloadManager = remember { ModelDownloadManager(context) }
    val orchestrator = remember { FluxPipelineOrchestrator(context) }

    var currentDestination by remember { mutableStateOf(AppDestination.STUDIO) }
    val downloadState by downloadManager.downloadState.collectAsState()
    var pipelineProgress by remember { mutableStateOf<PipelineProgress>(PipelineProgress.Idle) }

    var isModelReady by remember {
        mutableStateOf(ModelManifest.isFullyInstalled(downloadManager.getModelDirectory()))
    }

    // Refresh model installation status when switching tabs or when download state changes
    LaunchedEffect(currentDestination, downloadState) {
        isModelReady = ModelManifest.isFullyInstalled(downloadManager.getModelDirectory())
    }

    Scaffold(
        bottomBar = {
            NavigationBar {
                AppDestination.values().forEach { destination ->
                    NavigationBarItem(
                        selected = currentDestination == destination,
                        onClick = { currentDestination = destination },
                        icon = { Icon(destination.icon, contentDescription = destination.label) },
                        label = { Text(destination.label) },
                        colors = NavigationBarItemDefaults.colors(indicatorColor = FluxAccent.copy(alpha = 0.2f))
                    )
                }
            }
        }
    ) { innerPadding ->
        Surface(modifier = Modifier.padding(innerPadding)) {
            when (currentDestination) {
                AppDestination.STUDIO -> {
                    EditorScreen(
                        isModelReady = isModelReady,
                        progress = pipelineProgress,
                        onGenerate = { params: GenerationParams ->
                            coroutineScope.launch {
                                orchestrator.generate(params) { prog ->
                                    pipelineProgress = prog
                                }
                            }
                        },
                        onSaveToGallery = { bitmap ->
                            saveBitmapToGallery(context, bitmap)
                        },
                        onShareImage = { bitmap ->
                            shareBitmap(context, bitmap)
                        }
                    )
                }
                AppDestination.MODELS -> {
                    ModelManagerScreen(
                        baseDir = downloadManager.getModelDirectory(),
                        downloadState = downloadState,
                        onStartDownload = {
                            coroutineScope.launch {
                                downloadManager.downloadAllModels {
                                    isModelReady = true
                                }
                            }
                        },
                        onImportLocal = {
                            Toast.makeText(context, "Select folder with 43 model assets", Toast.LENGTH_SHORT).show()
                        }
                    )
                }
            }
        }
    }
}

fun saveBitmapToGallery(context: Context, bitmap: Bitmap) {
    try {
        val filename = "FluxEdit_${System.currentTimeMillis()}.png"
        var fos: OutputStream? = null

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val resolver = context.contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/FluxEdit")
            }
            val imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            if (imageUri != null) {
                fos = resolver.openOutputStream(imageUri)
            }
        } else {
            val imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString() + "/FluxEdit"
            val file = File(imagesDir)
            if (!file.exists()) file.mkdirs()
            val image = File(imagesDir, filename)
            fos = FileOutputStream(image)
        }

        fos?.use {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
            Toast.makeText(context, "Saved image to Gallery!", Toast.LENGTH_SHORT).show()
        }
    } catch (e: Exception) {
        Toast.makeText(context, "Failed to save: ${e.message}", Toast.LENGTH_SHORT).show()
    }
}

fun shareBitmap(context: Context, bitmap: Bitmap) {
    try {
        val cachePath = File(context.cacheDir, "images")
        cachePath.mkdirs()
        val file = File(cachePath, "flux_shared_image.png")
        val stream = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        stream.close()

        val contentUri: Uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)

        val shareIntent = Intent().apply {
            action = Intent.ACTION_SEND
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            setDataAndType(contentUri, context.contentResolver.getType(contentUri))
            putExtra(Intent.EXTRA_STREAM, contentUri)
            type = "image/png"
        }
        context.startActivity(Intent.createChooser(shareIntent, "Share edited image"))
    } catch (e: Exception) {
        Toast.makeText(context, "Share error: ${e.message}", Toast.LENGTH_SHORT).show()
    }
}

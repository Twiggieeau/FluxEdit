package com.gestura.fluxedit.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gestura.fluxedit.model.DownloadState
import com.gestura.fluxedit.model.ModelFileInfo
import com.gestura.fluxedit.model.ModelManifest
import com.gestura.fluxedit.ui.theme.FluxAccent
import com.gestura.fluxedit.ui.theme.FluxCardDark
import com.gestura.fluxedit.ui.theme.FluxSuccess
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelManagerScreen(
    baseDir: File,
    downloadState: DownloadState,
    onStartDownload: () -> Unit,
    onImportLocal: () -> Unit
) {
    val fileStatus = remember(downloadState) {
        ModelManifest.checkInstalledFiles(baseDir)
    }

    val installedCount = fileStatus.count { it.second }
    val totalCount = ModelManifest.REQUIRED_FILES.size
    val isAllReady = installedCount == totalCount

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Model Pack Manager", fontWeight = FontWeight.Bold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Summary Card
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = FluxCardDark)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                text = "FLUX.2 Klein 4B LiteRT",
                                fontWeight = FontWeight.Black,
                                fontSize = 16.sp
                            )
                            Text(
                                text = "HuggingFace: ${ModelManifest.REPO_ID}",
                                color = Color.Gray,
                                fontSize = 11.sp
                            )
                        }

                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = if (isAllReady) FluxSuccess.copy(alpha = 0.2f) else MaterialTheme.colorScheme.error.copy(alpha = 0.2f)
                        ) {
                            Text(
                                text = if (isAllReady) "READY" else "$installedCount / $totalCount FILES",
                                color = if (isAllReady) FluxSuccess else MaterialTheme.colorScheme.error,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Total Payload: 8.69 GB | Recommended RAM: 8 GB - 12 GB+",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    // Action buttons
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        val isDownloading = downloadState is DownloadState.Downloading || downloadState is DownloadState.Verifying

                        Button(
                            onClick = onStartDownload,
                            enabled = !isDownloading && !isAllReady,
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = FluxAccent)
                        ) {
                            if (isDownloading) {
                                CircularProgressIndicator(color = Color.White, modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Downloading...", fontSize = 12.sp)
                            } else {
                                Icon(Icons.Default.CloudDownload, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(if (isAllReady) "Downloaded" else "Download Assets", fontSize = 12.sp)
                            }
                        }

                        OutlinedButton(
                            onClick = onImportLocal,
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Import Files", fontSize = 12.sp)
                        }
                    }
                }
            }

            // Live Download Progress Card
            if (downloadState is DownloadState.Downloading) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Text(
                            text = "File ${downloadState.fileIndex}/${downloadState.totalFiles}: ${downloadState.currentFileName}",
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        LinearProgressIndicator(
                            progress = { downloadState.fileProgress },
                            modifier = Modifier.fillMaxWidth(),
                            color = FluxAccent
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "${(downloadState.totalBytesDownloaded / (1024 * 1024))} MB / ${(downloadState.totalBytes / (1024 * 1024))} MB",
                            color = Color.Gray,
                            fontSize = 11.sp
                        )
                    }
                }
            } else if (downloadState is DownloadState.Verifying) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(downloadState.currentFileName, fontSize = 13.sp)
                    }
                }
            }

            // File Checklist (43 items)
            Text("Contract Files Checklist (43)", fontWeight = FontWeight.Bold, fontSize = 14.sp)
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(fileStatus) { (info, isPresent) ->
                    FileStatusItem(info = info, isPresent = isPresent)
                }
            }
        }
    }
}

@Composable
fun FileStatusItem(info: ModelFileInfo, isPresent: Boolean) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = FluxCardDark)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = info.relativePath,
                    fontWeight = FontWeight.Medium,
                    fontSize = 12.sp,
                    maxLines = 1
                )
                Text(
                    text = "${info.group} • ${info.expectedSizeBytes / (1024 * 1024)} MB",
                    color = Color.Gray,
                    fontSize = 10.sp
                )
            }

            Icon(
                imageVector = if (isPresent) Icons.Default.CheckCircle else Icons.Default.Cancel,
                contentDescription = null,
                tint = if (isPresent) FluxSuccess else Color.Gray,
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

package com.gestura.fluxedit.ui.screens

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gestura.fluxedit.pipeline.GenerationParams
import com.gestura.fluxedit.pipeline.PipelineProgress
import com.gestura.fluxedit.pipeline.WorkflowType
import com.gestura.fluxedit.ui.theme.FluxAccent
import com.gestura.fluxedit.ui.theme.FluxCardDark
import com.gestura.fluxedit.ui.theme.FluxSuccess

data class GeometryOption(val label: String, val width: Int, val height: Int)

val GEOMETRY_OPTIONS = listOf(
    GeometryOption("512 × 512 (Square 1:1)", 512, 512),
    GeometryOption("512 × 384 (Landscape 4:3)", 512, 384),
    GeometryOption("384 × 512 (Portrait 3:4)", 384, 512),
    GeometryOption("512 × 288 (Widescreen 16:9)", 512, 288),
    GeometryOption("288 × 512 (Story 9:16)", 288, 512),
    GeometryOption("480 × 320 (Photo 3:2)", 480, 320),
    GeometryOption("320 × 480 (Photo 2:3)", 320, 480),
    GeometryOption("256 × 256 (Fast 1:1)", 256, 256)
)

val PRESET_PROMPTS = listOf(
    "Turn the red apple into a vibrant green apple with dew drops",
    "Add stylish retro sunglasses and cinematic neon rim lighting",
    "Transform the scene into a snowy winter wonderland",
    "Cyberpunk city street style with holographic reflections",
    "Make it a classical oil painting with textured brushstrokes",
    "Change the background into a tropical beach at sunset"
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EditorScreen(
    isModelReady: Boolean,
    progress: PipelineProgress,
    onGenerate: (GenerationParams) -> Unit,
    onSaveToGallery: (Bitmap) -> Unit,
    onShareImage: (Bitmap) -> Unit
) {
    val workflowTabs = remember {
        listOf(
            WorkflowType.SINGLE_IMAGE_EDIT,
            WorkflowType.TWO_IMAGE_EDIT,
            WorkflowType.TEXT_TO_IMAGE
        )
    }
    var selectedWorkflow by remember { mutableStateOf(WorkflowType.SINGLE_IMAGE_EDIT) }
    var prompt by remember { mutableStateOf("Turn the red apple into a vibrant green apple") }
    var initStrength by remember { mutableStateOf(1.0f) }
    var selectedGeometry by remember { mutableStateOf(GEOMETRY_OPTIONS[0]) }
    var randomSeed by remember { mutableStateOf(true) }
    var customSeed by remember { mutableStateOf(42L) }

    var image1Bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var image2Bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var generatedBitmap by remember { mutableStateOf<Bitmap?>(null) }

    // Split slider position (0f to 1f) for Before / After comparison
    var splitPosition by remember { mutableStateOf(0.5f) }

    val context = LocalContext.current

    // Launchers for picking images
    val image1Launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            val stream = context.contentResolver.openInputStream(it)
            image1Bitmap = android.graphics.BitmapFactory.decodeStream(stream)
        }
    }

    val image2Launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            val stream = context.contentResolver.openInputStream(it)
            image2Bitmap = android.graphics.BitmapFactory.decodeStream(stream)
        }
    }

    LaunchedEffect(progress) {
        if (progress is PipelineProgress.Completed) {
            generatedBitmap = progress.outputBitmap
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "FluxEdit",
                            fontWeight = FontWeight.Black,
                            fontSize = 22.sp
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Surface(
                            shape = RoundedCornerShape(6.dp),
                            color = FluxAccent.copy(alpha = 0.2f)
                        ) {
                            Text(
                                text = "FLUX.2 Klein 4B",
                                color = FluxAccent,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                        }
                    }
                },
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
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Workflow mode selector
            val selectedTabIndex = workflowTabs.indexOf(selectedWorkflow).coerceAtLeast(0)
            TabRow(
                selectedTabIndex = selectedTabIndex,
                containerColor = FluxCardDark,
                contentColor = MaterialTheme.colorScheme.onSurface
            ) {
                Tab(
                    selected = selectedWorkflow == WorkflowType.SINGLE_IMAGE_EDIT,
                    onClick = { selectedWorkflow = WorkflowType.SINGLE_IMAGE_EDIT },
                    text = { Text("Image Edit", fontSize = 13.sp) },
                    icon = { Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(18.dp)) }
                )
                Tab(
                    selected = selectedWorkflow == WorkflowType.TWO_IMAGE_EDIT,
                    onClick = { selectedWorkflow = WorkflowType.TWO_IMAGE_EDIT },
                    text = { Text("Blend (2 Img)", fontSize = 13.sp) },
                    icon = { Icon(Icons.Default.Layers, contentDescription = null, modifier = Modifier.size(18.dp)) }
                )
                Tab(
                    selected = selectedWorkflow == WorkflowType.TEXT_TO_IMAGE,
                    onClick = { selectedWorkflow = WorkflowType.TEXT_TO_IMAGE },
                    text = { Text("Text-to-Image", fontSize = 13.sp) },
                    icon = { Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(18.dp)) }
                )
            }

            // Canvas & Preview Area
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = FluxCardDark)
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    val result = generatedBitmap
                    val original = image1Bitmap

                    if (result != null && original != null && selectedWorkflow != WorkflowType.TEXT_TO_IMAGE) {
                        // Before / After Comparison View
                        BoxWithConstraints(
                            modifier = Modifier
                                .fillMaxSize()
                                .pointerInput(Unit) {
                                    detectHorizontalDragGestures { _, dragAmount ->
                                        val newPos = splitPosition + (dragAmount / size.width)
                                        splitPosition = newPos.coerceIn(0.05f, 0.95f)
                                    }
                                }
                        ) {
                            // After image (full width, visible on right side)
                            Image(
                                bitmap = result.asImageBitmap(),
                                contentDescription = "Edited result",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Fit
                            )

                            // Before image (clipped to left of splitPosition)
                            Image(
                                bitmap = original.asImageBitmap(),
                                contentDescription = "Original image",
                                modifier = Modifier
                                    .fillMaxSize()
                                    .drawWithContent {
                                        clipRect(right = size.width * splitPosition) {
                                            this@drawWithContent.drawContent()
                                        }
                                    },
                                contentScale = ContentScale.Fit
                            )

                            // Split divider bar
                            Box(
                                modifier = Modifier
                                    .fillMaxHeight()
                                    .width(3.dp)
                                    .align(Alignment.CenterStart)
                                    .offset(x = maxWidth * splitPosition)
                                    .background(Color.White)
                            )

                            // Comparison labels
                            Surface(
                                shape = RoundedCornerShape(4.dp),
                                color = Color.Black.copy(alpha = 0.7f),
                                modifier = Modifier.padding(8.dp).align(Alignment.TopStart)
                            ) {
                                Text("Original", color = Color.White, fontSize = 11.sp, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp))
                            }
                            Surface(
                                shape = RoundedCornerShape(4.dp),
                                color = FluxAccent.copy(alpha = 0.8f),
                                modifier = Modifier.padding(8.dp).align(Alignment.TopEnd)
                            ) {
                                Text("FLUX Edit", color = Color.White, fontSize = 11.sp, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp))
                            }
                        }
                    } else if (result != null) {
                        // Generated result
                        Image(
                            bitmap = result.asImageBitmap(),
                            contentDescription = "Generated Image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } else if (original != null && selectedWorkflow != WorkflowType.TEXT_TO_IMAGE) {
                        Image(
                            bitmap = original.asImageBitmap(),
                            contentDescription = "Original image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } else {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.AddPhotoAlternate,
                                contentDescription = null,
                                modifier = Modifier.size(48.dp),
                                tint = Color.Gray
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = if (selectedWorkflow == WorkflowType.TEXT_TO_IMAGE) "Ready to synthesize" else "Select an image to start editing",
                                color = Color.Gray,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }

            // Image Slot Buttons (for 1-Image and 2-Image workflows)
            if (selectedWorkflow != WorkflowType.TEXT_TO_IMAGE) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = { image1Launcher.launch("image/*") },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.Photo, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(if (image1Bitmap != null) "Change Image 1" else "Pick Image 1", fontSize = 12.sp)
                    }

                    if (selectedWorkflow == WorkflowType.TWO_IMAGE_EDIT) {
                        OutlinedButton(
                            onClick = { image2Launcher.launch("image/*") },
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Icon(Icons.Default.Filter, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(if (image2Bitmap != null) "Change Image 2" else "Pick Image 2", fontSize = 12.sp)
                        }
                    }
                }
            }

            // Prompt Input Card
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = FluxCardDark)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Instruction / Prompt", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = prompt,
                        onValueChange = { prompt = it },
                        modifier = Modifier.fillMaxWidth(),
                        placeholder = { Text("e.g. Turn the red apple into a green apple") },
                        maxLines = 3,
                        shape = RoundedCornerShape(12.dp)
                    )

                    Spacer(modifier = Modifier.height(12.dp))
                    Text("Preset Ideas", color = Color.Gray, fontSize = 12.sp)
                    Spacer(modifier = Modifier.height(6.dp))
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(PRESET_PROMPTS) { preset ->
                            SuggestionChip(
                                onClick = { prompt = preset },
                                label = { Text(preset, fontSize = 11.sp, maxLines = 1) }
                            )
                        }
                    }
                }
            }

            // Parameters & Geometry Card
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = FluxCardDark)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    // Geometry selection
                    Text("Target Resolution (divisible by 32)", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Spacer(modifier = Modifier.height(8.dp))

                    var expandedGeo by remember { mutableStateOf(false) }
                    ExposedDropdownMenuBox(
                        expanded = expandedGeo,
                        onExpandedChange = { expandedGeo = !expandedGeo }
                    ) {
                        OutlinedTextField(
                            value = selectedGeometry.label,
                            onValueChange = {},
                            readOnly = true,
                            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedGeo) },
                            modifier = Modifier.menuAnchor().fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp)
                        )
                        ExposedDropdownMenu(
                            expanded = expandedGeo,
                            onDismissRequest = { expandedGeo = false }
                        ) {
                            GEOMETRY_OPTIONS.forEach { opt ->
                                DropdownMenuItem(
                                    text = { Text(opt.label) },
                                    onClick = {
                                        selectedGeometry = opt
                                        expandedGeo = false
                                    }
                                )
                            }
                        }
                    }

                    // Strength Slider (for image editing)
                    if (selectedWorkflow != WorkflowType.TEXT_TO_IMAGE) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Edit Strength (initStrength)", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                            Text("${(initStrength * 100).toInt()}%", color = FluxAccent, fontWeight = FontWeight.Bold)
                        }
                        Slider(
                            value = initStrength,
                            onValueChange = { initStrength = it },
                            valueRange = 0.1f..1.0f,
                            colors = SliderDefaults.colors(thumbColor = FluxAccent, activeTrackColor = FluxAccent)
                        )
                        Text(
                            text = "Low = subtle modification; High = bold transformation",
                            color = Color.Gray,
                            fontSize = 11.sp
                        )
                    }

                    // Seed Settings
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(
                                checked = randomSeed,
                                onCheckedChange = { randomSeed = it }
                            )
                            Text("Random Seed", fontSize = 13.sp)
                        }
                        if (!randomSeed) {
                            OutlinedTextField(
                                value = customSeed.toString(),
                                onValueChange = { customSeed = it.toLongOrNull() ?: 0L },
                                modifier = Modifier.width(120.dp),
                                singleLine = true,
                                shape = RoundedCornerShape(8.dp)
                            )
                        }
                    }
                }
            }

            // Progress status display
            AnimatedVisibility(visible = progress !is PipelineProgress.Idle) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        val statusText = when (progress) {
                            is PipelineProgress.Tokenizing -> "Formatting prompt chat template & tokenizing..."
                            is PipelineProgress.EncodingText -> "Running 3-stage text encoder (ke_enc0..2)..."
                            is PipelineProgress.EncodingImages -> "Encoding image ${progress.current}/${progress.total} via VAE..."
                            is PipelineProgress.Denoising -> "Denoising step ${progress.step}/${progress.totalSteps} (${progress.stageName})..."
                            is PipelineProgress.DecodingVae -> "Decoding VAE latents to output RGB image..."
                            is PipelineProgress.Completed -> "Generation complete in ${progress.durationMs / 1000.0}s!"
                            is PipelineProgress.Error -> "Error: ${progress.message}"
                            else -> ""
                        }

                        Text(statusText, fontWeight = FontWeight.Medium, fontSize = 13.sp)
                        Spacer(modifier = Modifier.height(8.dp))

                        val progressFraction = when (progress) {
                            is PipelineProgress.Tokenizing -> 0.1f
                            is PipelineProgress.EncodingText -> 0.25f
                            is PipelineProgress.EncodingImages -> 0.4f
                            is PipelineProgress.Denoising -> 0.4f + progress.percent * 0.5f
                            is PipelineProgress.DecodingVae -> 0.95f
                            is PipelineProgress.Completed -> 1.0f
                            else -> 0f
                        }

                        LinearProgressIndicator(
                            progress = { progressFraction },
                            modifier = Modifier.fillMaxWidth(),
                            color = if (progress is PipelineProgress.Error) MaterialTheme.colorScheme.error else FluxAccent
                        )
                    }
                }
            }

            // Main Action Button
            val isGenerating = progress !is PipelineProgress.Idle && progress !is PipelineProgress.Completed && progress !is PipelineProgress.Error
            val hasRequiredImages = when (selectedWorkflow) {
                WorkflowType.TEXT_TO_IMAGE -> true
                WorkflowType.SINGLE_IMAGE_EDIT -> image1Bitmap != null
                WorkflowType.TWO_IMAGE_EDIT -> image1Bitmap != null && image2Bitmap != null
            }

            Button(
                onClick = {
                    val finalSeed = if (randomSeed) System.currentTimeMillis() else customSeed
                    onGenerate(
                        GenerationParams(
                            prompt = prompt,
                            workflow = selectedWorkflow,
                            image1 = image1Bitmap,
                            image2 = image2Bitmap,
                            width = selectedGeometry.width,
                            height = selectedGeometry.height,
                            seed = finalSeed,
                            initStrength = initStrength
                        )
                    )
                },
                enabled = isModelReady && !isGenerating && hasRequiredImages,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(containerColor = FluxAccent)
            ) {
                if (isGenerating) {
                    CircularProgressIndicator(color = Color.White, modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(10.dp))
                    Text("Generating (4 Steps)...", fontWeight = FontWeight.Bold)
                } else {
                    Icon(Icons.Default.AutoFixHigh, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = when (selectedWorkflow) {
                            WorkflowType.SINGLE_IMAGE_EDIT -> "Generate Image Edit"
                            WorkflowType.TWO_IMAGE_EDIT -> "Synthesize 2-Image Blend"
                            WorkflowType.TEXT_TO_IMAGE -> "Synthesize Image"
                        },
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // Export Actions (Save & Share)
            generatedBitmap?.let { bmp ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Button(
                        onClick = { onSaveToGallery(bmp) },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = FluxSuccess),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.SaveAlt, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Save to Photos")
                    }

                    OutlinedButton(
                        onClick = { onShareImage(bmp) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Share")
                    }
                }
            }
        }
    }
}

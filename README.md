# Gestura FluxEdit: On-Device FLUX.2 Klein 4B Android Image Editor

**FluxEdit** is a high-performance, on-device Android image editing and synthesis application built specifically for the **[Gestura FLUX.2 Klein 4B LiteRT/TFLite](https://huggingface.co/ZawShiShawn/gestura-flux2-klein-4b-litert-tflite)** runtime model package.

The application allows users to edit existing images via natural language instructions, blend multiple images together, or generate new images directly on their Android devices with 4-step shifted-flow Euler denoising.

---

## Key Features

1. **Instruction-Based Image Editing (`kce_*` family)**:
   - Modify existing photos with prompts (e.g. *"turn the red apple into a green apple"*, *"add retro sunglasses and neon lighting"*).
   - Adjustable `initStrength` slider (0.0 to 1.0) controlling editing magnitude.
   - Preserves conditioning image latent at temporal position `10` while updating noisy latent at temporal position `0`.

2. **Two-Image Composition & Blending (`kce2_*` family)**:
   - Combines subjects or styles from two separate images guided by text instructions.
   - Binds Image 1 at temporal position `10` and Image 2 at temporal position `20`.

3. **Text-to-Image Generation (`kc_*` family)**:
   - Generates images from pure Gaussian noise seeds using 4-step shifted-flow scheduler.

4. **Interactive Studio UI (Jetpack Compose & Material 3)**:
   - **Interactive Before / After Comparison Slider**: Drag horizontally across the screen to view original vs edited image in real time.
   - **Resolution & Aspect Ratio Selector**: 512×512 (1:1), 512×384 (4:3), 384×512 (3:4), 512×288 (16:9), 288×512 (9:16), 480×320 (3:2), 320×480 (2:3), 256×256.
   - **Real-Time Stage Progress Bar**: Tokenizing → Text Encoding → VAE Encoding → 4-Step Denoising Loop → VAE Decoding.
   - **Export & Share**: Save directly to Android Gallery (`Pictures/FluxEdit`) or share via Android Intent.

5. **Integrated Model Pack Manager**:
   - Manages the 43 payload files (8.69 GB total).
   - Resumable downloading directly from HuggingFace Hub (`ZawShiShawn/gestura-flux2-klein-4b-litert-tflite`).
   - SHA-256 integrity verification against contract specifications for external weight sidecars.
   - Local directory import for offline transfers.

---

## Architecture & Contract Compliance

```
i2i/
├── app/
│   ├── build.gradle.kts
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml
│       │   ├── cpp/                       # Native C++ LiteRT Engine (libflux_engine.so)
│       │   │   ├── CMakeLists.txt
│       │   │   ├── flux_engine.h          # RoPE, Latent Patchify/Unpack, Euler step
│       │   │   ├── flux_engine.cpp        # External weight mmap, BF16/FP16 vector math
│       │   │   └── jni_bridge.cpp         # JNI bridge to Kotlin
│       │   ├── java/com/gestura/fluxedit/
│       │   │   ├── engine/
│       │   │   │   └── FluxNativeEngine.kt
│       │   │   ├── model/
│       │   │   │   ├── ModelManifest.kt   # 43-file contract specifications & SHA-256
│       │   │   │   └── ModelDownloadManager.kt # Resumable HuggingFace downloader
│       │   │   ├── pipeline/
│       │   │   │   ├── TextEncoderRunner.kt   # 3-stage ke_enc0..2 + 7680-dim conditioning
│       │   │   │   ├── VaePipeline.kt         # VAE encode/decode, planar RGB [-1, 1]
│       │   │   │   └── FluxPipelineOrchestrator.kt # 4-step Euler denoising loop
│       │   │   ├── tokenizer/
│       │   │   │   └── QwenBpeTokenizer.kt    # Qwen2 BPE + exact chat template
│       │   │   └── ui/
│       │   │       ├── MainActivity.kt
│       │   │       ├── FluxEditApp.kt
│       │   │       ├── screens/
│       │   │       │   ├── EditorScreen.kt    # Studio with Before/After slider
│       │   │       │   └── ModelManagerScreen.kt # Model asset status dashboard
│       │   │       └── theme/
│       │   └── res/                       # Material 3 colors, strings, adaptive icon
│       └── test/                          # Unit tests for tokenizer, manifest, & Euler math
├── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

---

## Hardware Requirements

- **Processor**: ARM64-v8a (Qualcomm Snapdragon 8 Gen 1 / 8 Gen 2 / 8 Gen 3 or Samsung Exynos 2200 / 2400+).
- **RAM**: Minimum 8 GB RAM (12 GB+ recommended for optimal latency).
- **Storage**: ~10 GB free internal storage for model files.

---

## Building the Project

Ensure you have Android Studio Ladybug or later, Android SDK 35, and NDK 27+.

```bash
# Clone or navigate to the project directory
cd i2i

# Build debug APK
./gradlew assembleDebug

# Run unit tests
./gradlew test
```

---

## Attribution & License

- Based on **FLUX.2 [klein] 4B** by Black Forest Labs, licensed under the **Apache License 2.0**.
- Associated with the converted LiteRT/TFLite Android package by **ZawShiShawn/gestura-flux2-klein-4b-litert-tflite**.
- Text encoder based on Qwen3-4B / Alibaba Cloud architecture.

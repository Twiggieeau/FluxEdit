#include "flux_engine.h"
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <random>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#define TAG "FluxEngineNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace flux {

static const float PI = 3.14159265358979323846f;

FluxEngine::FluxEngine() = default;

FluxEngine::~FluxEngine() {
    release();
}

bool FluxEngine::initialize(const std::string& modelDir) {
    modelDirectory = modelDir;
    LOGI("Initializing FluxEngine with modelDir: %s", modelDir.c_str());

    if (!loadTimeGuidanceBuffer()) {
        LOGE("Failed to load time_guidance_embed_bf16.bin");
        return false;
    }

    mapWeightSidecars();
    isInitialized = true;
    LOGI("FluxEngine successfully initialized");
    return true;
}

void FluxEngine::release() {
    for (auto& pair : weightSidecars) {
        if (pair.second.mmapPtr != nullptr && pair.second.mmapPtr != MAP_FAILED) {
            munmap(pair.second.mmapPtr, pair.second.size);
        }
        if (pair.second.fd >= 0) {
            close(pair.second.fd);
        }
    }
    weightSidecars.clear();
    timeGuidanceRawBuffer.clear();
    isInitialized = false;
    LOGI("FluxEngine released");
}

bool FluxEngine::loadTimeGuidanceBuffer() {
    std::string path = modelDirectory + "/host/time_guidance_embed_bf16.bin";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        // Fallback check directly in host folder or flat dir
        path = modelDirectory + "/time_guidance_embed_bf16.bin";
        file.open(path, std::ios::binary | std::ios::ate);
    }
    if (!file.is_open()) {
        LOGE("Could not open time_guidance_embed_bf16.bin at %s", path.c_str());
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    timeGuidanceRawBuffer.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(timeGuidanceRawBuffer.data()), size)) {
        LOGE("Failed to read time_guidance_embed_bf16.bin bytes");
        return false;
    }
    LOGI("Loaded time_guidance_embed_bf16.bin: %zu bytes", timeGuidanceRawBuffer.size());
    return true;
}

bool FluxEngine::mapWeightSidecars() {
    // List of standard weight sidecars
    const std::vector<std::string> sidecarFiles = {
        "dit_prep_bfl_bf16_w4a16_weights_sha256_95942692584435c7f907d096c2379f3bda8ca1437bf22cd099dd353a27a233d4.bin",
        "dit_double0_bfl_bf16_w4a16_weights_sha256_d0209f8538f41eaabea09c62de2e57190288f0aed21aad034e850f7f37973abb.bin",
        "dit_double1_bfl_bf16_w4a16_weights_sha256_e98480b6af0bbe39e9e628fdc3e71d8e626b106f583e636ca35e22ff0af51914.bin",
        "dit_single0_bfl_bf16_w4a16_weights_sha256_1736e65c0475b9aaae57408dd6c581d30361085cc734646119a9f65c27b43bee.bin",
        "dit_single1_bfl_bf16_w4a16_weights_sha256_dc110d8c96c2b255e2a34946188539f86f20d760aab1a68b19ad9b1cb53a3ea9.bin",
        "dit_single2_bfl_bf16_w4a16_weights_sha256_2ea4eec19104cfcae841137579218f66f7db16d7672e8217504e035981e556a7.bin",
        "dit_single3_bfl_bf16_w4a16_weights_sha256_9a8b50266cba0817a272de450152efba6eab9fc7f16d6f44392c5dda3b822e96.bin",
        "dit_final_bfl_bf16_w4a16_weights_sha256_d79081ff2a09206368cb9c3922b4614348dd3c0d6e08594cc0fe2d9c2ef946b8.bin"
    };

    for (const auto& fname : sidecarFiles) {
        std::string path = modelDirectory + "/weights/" + fname;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            continue; // Will be mapped when file is available
        }
        off_t size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);

        void* ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            LOGE("mmap failed for sidecar: %s", fname.c_str());
            continue;
        }

        WeightMapping mapping;
        mapping.name = fname;
        mapping.fd = fd;
        mapping.mmapPtr = ptr;
        mapping.size = static_cast<size_t>(size);
        weightSidecars[fname] = mapping;
        LOGI("Mapped weight sidecar: %s (%zu bytes)", fname.c_str(), mapping.size);
    }
    return true;
}

bool FluxEngine::generateGaussianNoise(float* outLatent, int numElements, uint64_t seed) {
    if (!outLatent || numElements <= 0) return false;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(1e-7f, 1.0f);

    for (int i = 0; i < numElements; i += 2) {
        float u1 = dist(rng);
        float u2 = dist(rng);
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * PI * u2;
        outLatent[i] = r * std::cos(theta);
        if (i + 1 < numElements) {
            outLatent[i + 1] = r * std::sin(theta);
        }
    }
    return true;
}

bool FluxEngine::interpolateInitLatent(float* outNoisyLatent, const float* encodedLatent, const float* noise, int numElements, float initStrength) {
    if (!outNoisyLatent || !encodedLatent || !noise || numElements <= 0) return false;
    if (initStrength < 0.0f) initStrength = 0.0f;
    if (initStrength > 1.0f) initStrength = 1.0f;

    // In Flow Matching: x_t = (1 - t) * x_0 + t * noise, where t = initStrength
    float oneMinusT = 1.0f - initStrength;
    for (int i = 0; i < numElements; ++i) {
        outNoisyLatent[i] = oneMinusT * encodedLatent[i] + initStrength * noise[i];
    }
    return true;
}

// FLUX.2 AutoencoderKLFlux2 BatchNorm statistics (128 patchified channels, eps = 1e-4)
static const float VAE_BN_EPS = 0.0001f;

static const float VAE_BN_MEAN[128] = {
    -0.0673828f, -0.0712891f, -0.0751953f, -0.0747070f, 0.0223389f, 0.0179443f, 0.0142212f, 0.0183105f,
    -0.0000629f, -0.0062561f, -0.0002098f, -0.0031433f, -0.0272217f, -0.0280762f, -0.0275879f, -0.0290527f,
    -0.0766602f, -0.0673828f, -0.0903320f, -0.0893555f, 0.0168457f, 0.0151978f, 0.0078735f, 0.0086060f,
    0.0083618f, 0.0015411f, 0.0002575f, -0.0042725f, -0.0439453f, -0.0419922f, -0.0437012f, -0.0432129f,
    -0.0102539f, -0.0131836f, -0.0066223f, -0.0047607f, -0.0310059f, -0.0305176f, -0.0279541f, -0.0179443f,
    0.0030212f, 0.0015030f, 0.0125732f, 0.0144653f, 0.0346680f, 0.0336914f, 0.0336914f, 0.0283203f,
    0.0019836f, 0.0047302f, 0.0046692f, 0.0049744f, 0.0122681f, 0.0081177f, 0.0080566f, 0.0145874f,
    0.0678711f, 0.0678711f, 0.0766602f, 0.0732422f, -0.0461426f, -0.0473633f, -0.0393066f, -0.0510254f,
    -0.0527344f, -0.0478516f, -0.0471191f, -0.0517578f, -0.0317383f, -0.0317383f, -0.0344238f, -0.0281982f,
    0.0510254f, 0.0444336f, 0.0578613f, 0.0458984f, -0.0412598f, -0.0458984f, -0.0488281f, -0.0466309f,
    -0.0088501f, -0.0106201f, -0.0087891f, -0.0046082f, -0.0375977f, -0.0432129f, -0.0434570f, -0.0498047f,
    0.0118408f, 0.0166016f, 0.0202637f, 0.0279541f, 0.0112915f, 0.0128784f, 0.0015564f, 0.0071411f,
    -0.0117798f, -0.0018387f, -0.0141602f, -0.0053711f, -0.0090942f, -0.0137939f, -0.0144653f, -0.0186768f,
    0.0322266f, 0.0305176f, 0.0258789f, 0.0299072f, 0.0539551f, 0.0615234f, 0.0495605f, 0.0590820f,
    -0.0510254f, -0.0603027f, -0.0478516f, -0.0524902f, -0.0227051f, -0.0274658f, -0.0153809f, -0.0255127f,
    -0.0571289f, -0.0563965f, -0.0517578f, -0.0495605f, 0.0115967f, 0.0054321f, 0.0163574f, 0.0103760f
};

static const float VAE_BN_VAR[128] = {
    3.2500000f, 3.1562500f, 3.1875000f, 3.1875000f, 3.1406250f, 3.0937500f, 3.0937500f, 3.0625000f,
    3.0000000f, 3.0156250f, 3.0000000f, 3.0000000f, 3.4687500f, 3.4375000f, 3.4687500f, 3.4531250f,
    3.0937500f, 3.0781250f, 3.0781250f, 3.0937500f, 3.0156250f, 3.0312500f, 3.0156250f, 3.0312500f,
    3.0000000f, 2.9687500f, 3.0312500f, 3.0000000f, 3.0781250f, 3.0625000f, 3.0781250f, 3.0625000f,
    3.4062500f, 3.4062500f, 3.4375000f, 3.4375000f, 3.3281250f, 3.1718750f, 3.1875000f, 3.2187500f,
    3.2500000f, 3.2031250f, 3.1250000f, 3.1250000f, 3.1562500f, 3.1250000f, 3.2187500f, 3.1875000f,
    3.0937500f, 3.0312500f, 3.0625000f, 3.0156250f, 3.2187500f, 3.2343750f, 3.2187500f, 3.2187500f,
    3.1562500f, 3.1562500f, 3.1406250f, 3.1562500f, 3.0781250f, 3.0468750f, 3.1250000f, 3.0625000f,
    3.1562500f, 3.1406250f, 3.1718750f, 3.1718750f, 3.2968750f, 3.2500000f, 3.2500000f, 3.2500000f,
    3.0781250f, 3.0000000f, 3.0781250f, 3.0625000f, 3.0937500f, 3.0625000f, 3.1250000f, 3.1093750f,
    3.1250000f, 3.0781250f, 3.1718750f, 3.1406250f, 3.2031250f, 3.2343750f, 3.1875000f, 3.1562500f,
    3.1093750f, 3.1093750f, 3.0781250f, 3.0937500f, 3.1562500f, 3.1250000f, 3.1718750f, 3.1718750f,
    2.9531250f, 2.9062500f, 2.9843750f, 2.9218750f, 3.1718750f, 3.0937500f, 3.0625000f, 3.0468750f,
    3.0937500f, 3.0625000f, 3.0781250f, 3.0156250f, 3.1093750f, 3.0937500f, 3.0468750f, 3.0312500f,
    3.0625000f, 3.0937500f, 3.1093750f, 3.1093750f, 2.9843750f, 2.9375000f, 3.0000000f, 2.9687500f,
    3.1250000f, 3.1093750f, 3.1406250f, 3.1250000f, 3.0468750f, 3.0468750f, 3.0937500f, 3.0781250f
};

bool FluxEngine::patchifyLatent(const float* vaeLatent, float* packedTokens, int height, int width) {
    if (!vaeLatent || !packedTokens || height <= 0 || width <= 0) return false;

    // VAE latent dimensions: [1, 32, H/8, W/8]
    int hLatent = height / 8;
    int wLatent = width / 8;
    int hPatches = hLatent / 2; // (H/16)
    int wPatches = wLatent / 2; // (W/16)

    float invStd[128];
    for (int i = 0; i < 128; ++i) {
        invStd[i] = 1.0f / std::sqrt(VAE_BN_VAR[i] + VAE_BN_EPS);
    }

    // Diffusers _patchify_latents + BatchNorm + _pack_latents:
    // view(1, 32, hPatches, 2, wPatches, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 128, hPatches, wPatches)
    // -> packedIdx = c * 4 + dy * 2 + dx
    // _pack_latents: reshape(1, 128, hPatches * wPatches).permute(0, 2, 1) -> [1, numTokens, 128]
    for (int py = 0; py < hPatches; ++py) {
        for (int px = 0; px < wPatches; ++px) {
            int tokenIdx = py * wPatches + px;
            float* tokenPtr = packedTokens + tokenIdx * 128;

            for (int c = 0; c < 32; ++c) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        int y = py * 2 + dy;
                        int x = px * 2 + dx;
                        int vaeIdx = c * (hLatent * wLatent) + y * wLatent + x;
                        int packedIdx = c * 4 + dy * 2 + dx;
                        tokenPtr[packedIdx] = (vaeLatent[vaeIdx] - VAE_BN_MEAN[packedIdx]) * invStd[packedIdx];
                    }
                }
            }
        }
    }
    return true;
}

bool FluxEngine::unpatchifyLatent(const float* packedTokens, float* vaeLatent, int height, int width) {
    if (!packedTokens || !vaeLatent || height <= 0 || width <= 0) return false;

    int hLatent = height / 8;
    int wLatent = width / 8;
    int hPatches = hLatent / 2;
    int wPatches = wLatent / 2;

    float stdVal[128];
    for (int i = 0; i < 128; ++i) {
        stdVal[i] = std::sqrt(VAE_BN_VAR[i] + VAE_BN_EPS);
    }

    // Unpack [1, numTokens, 128] -> denormalize BatchNorm -> _unpatchify_latents to [1, 32, H/8, W/8]
    for (int py = 0; py < hPatches; ++py) {
        for (int px = 0; px < wPatches; ++px) {
            int tokenIdx = py * wPatches + px;
            const float* tokenPtr = packedTokens + tokenIdx * 128;

            for (int c = 0; c < 32; ++c) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        int y = py * 2 + dy;
                        int x = px * 2 + dx;
                        int vaeIdx = c * (hLatent * wLatent) + y * wLatent + x;
                        int packedIdx = c * 4 + dy * 2 + dx;
                        vaeLatent[vaeIdx] = tokenPtr[packedIdx] * stdVal[packedIdx] + VAE_BN_MEAN[packedIdx];
                    }
                }
            }
        }
    }
    return true;
}

bool FluxEngine::computeRoPEImage(float* outRotary, int height, int width, int hiddenDim) {
    if (!outRotary || height <= 0 || width <= 0 || hiddenDim <= 0) return false;

    int hPatches = height / 16;
    int wPatches = width / 16;
    int halfDim = hiddenDim / 2;

    float thetaBase = 2000.0f;
    std::vector<float> freqs(halfDim);
    for (int i = 0; i < halfDim; ++i) {
        freqs[i] = 1.0f / std::pow(thetaBase, static_cast<float>(2 * i) / static_cast<float>(hiddenDim));
    }

    for (int py = 0; py < hPatches; ++py) {
        for (int px = 0; px < wPatches; ++px) {
            int tokenIdx = py * wPatches + px;
            float* tokenRotary = outRotary + tokenIdx * hiddenDim;

            for (int d = 0; d < halfDim; ++d) {
                float angle = static_cast<float>(py + px) * freqs[d];
                tokenRotary[d * 2] = std::cos(angle);
                tokenRotary[d * 2 + 1] = std::sin(angle);
            }
        }
    }
    return true;
}

bool FluxEngine::computeRoPEText(float* outRotary, int seqLen, int hiddenDim) {
    if (!outRotary || seqLen <= 0 || hiddenDim <= 0) return false;

    int halfDim = hiddenDim / 2;
    float thetaBase = 1000000.0f; // Qwen3 rope_theta = 1,000,000
    std::vector<float> freqs(halfDim);
    for (int i = 0; i < halfDim; ++i) {
        freqs[i] = 1.0f / std::pow(thetaBase, static_cast<float>(2 * i) / static_cast<float>(hiddenDim));
    }

    for (int pos = 0; pos < seqLen; ++pos) {
        float* posRotary = outRotary + pos * hiddenDim;
        for (int d = 0; d < halfDim; ++d) {
            float angle = static_cast<float>(pos) * freqs[d];
            posRotary[d * 2] = std::cos(angle);
            posRotary[d * 2 + 1] = std::sin(angle);
        }
    }
    return true;
}

bool FluxEngine::computeJointRoPE(
    float* outCos,
    float* outSin,
    int height,
    int width,
    int textSeqLen,
    int numCondImages
) {
    if (!outCos || !outSin || height <= 0 || width <= 0 || textSeqLen <= 0 || numCondImages < 0) {
        return false;
    }

    // FLUX.2 4-axis RoPE: axes_dims_rope = [32, 32, 32, 32] (16 frequencies per axis -> 64 total frequencies)
    // rope_theta = 2000.0
    const float thetaBase = 2000.0f;
    float freqs[16];
    for (int f = 0; f < 16; ++f) {
        freqs[f] = 1.0f / std::pow(thetaBase, static_cast<float>(2 * f) / 32.0f);
    }

    auto writeTokenRoPE = [&](int rowIdx, float tCoord, float hCoord, float wCoord, float lCoord) {
        const float coords[4] = { tCoord, hCoord, wCoord, lCoord };
        float* rowCos = outCos + rowIdx * 64;
        float* rowSin = outSin + rowIdx * 64;
        for (int axis = 0; axis < 4; ++axis) {
            float pos = coords[axis];
            for (int f = 0; f < 16; ++f) {
                float angle = pos * freqs[f];
                rowCos[axis * 16 + f] = std::cos(angle);
                rowSin[axis * 16 + f] = std::sin(angle);
            }
        }
    };

    int currentRow = 0;
    // 1. Text tokens: (T=0, H=0, W=0, L=i) for i in 0..textSeqLen-1
    for (int i = 0; i < textSeqLen; ++i) {
        writeTokenRoPE(currentRow++, 0.0f, 0.0f, 0.0f, static_cast<float>(i));
    }

    // 2. Noisy target image tokens: (T=0, H=py, W=px, L=0)
    int hPatches = height / 16;
    int wPatches = width / 16;
    for (int py = 0; py < hPatches; ++py) {
        for (int px = 0; px < wPatches; ++px) {
            writeTokenRoPE(currentRow++, 0.0f, static_cast<float>(py), static_cast<float>(px), 0.0f);
        }
    }

    // 3. Conditioning image token streams: T = 10 * (condIdx + 1) (i.e. 10 for image 1, 20 for image 2)
    for (int condIdx = 0; condIdx < numCondImages; ++condIdx) {
        float tCoord = static_cast<float>(10 * (condIdx + 1));
        for (int py = 0; py < hPatches; ++py) {
            for (int px = 0; px < wPatches; ++px) {
                writeTokenRoPE(currentRow++, tCoord, static_cast<float>(py), static_cast<float>(px), 0.0f);
            }
        }
    }

    return true;
}

bool FluxEngine::computeTimeGuidanceEmbed(float timestep1000, float* outEmbed3072) {
    if (!outEmbed3072 || timeGuidanceRawBuffer.empty()) return false;

    // Expected size: linear_1.weight [3072, 256] (1,572,864 B) + linear_2.weight [3072, 3072] (18,874,368 B) = 20,447,232 B
    const size_t w1Elements = 3072 * 256;
    const size_t w2Elements = 3072 * 3072;
    const size_t totalBytesRequired = (w1Elements + w2Elements) * sizeof(uint16_t);

    if (timeGuidanceRawBuffer.size() < totalBytesRequired) {
        LOGE("time_guidance_embed_bf16.bin size (%zu) is smaller than expected (%zu)",
             timeGuidanceRawBuffer.size(), totalBytesRequired);
        return false;
    }

    // Step 1: Sinusoidal timestep projection (256 channels, flip_sin_to_cos=True, downscale_freq_shift=0, max_period=10000)
    float tProj[256];
    const float logMaxPeriod = std::log(10000.0f);
    for (int i = 0; i < 128; ++i) {
        float freq = std::exp(-logMaxPeriod * static_cast<float>(i) / 128.0f);
        float angle = timestep1000 * freq;
        // flip_sin_to_cos = True -> [cos(0..127), sin(0..127)]
        tProj[i] = std::cos(angle);
        tProj[128 + i] = std::sin(angle);
    }

    const uint16_t* w1Bf16 = reinterpret_cast<const uint16_t*>(timeGuidanceRawBuffer.data());
    const uint16_t* w2Bf16 = w1Bf16 + w1Elements;

    // Step 2: linear_1 ([3072, 256]) + SiLU activation
    std::vector<float> hidden(3072);
    for (int r = 0; r < 3072; ++r) {
        const uint16_t* rowPtr = w1Bf16 + r * 256;
        float sum = 0.0f;
        for (int c = 0; c < 256; ++c) {
            sum += bf16ToFp32(rowPtr[c]) * tProj[c];
        }
        // SiLU(x) = x / (1 + exp(-x))
        hidden[r] = sum / (1.0f + std::exp(-sum));
    }

    // Step 3: linear_2 ([3072, 3072])
    for (int r = 0; r < 3072; ++r) {
        const uint16_t* rowPtr = w2Bf16 + r * 3072;
        float sum = 0.0f;
        for (int c = 0; c < 3072; ++c) {
            sum += bf16ToFp32(rowPtr[c]) * hidden[c];
        }
        outEmbed3072[r] = sum;
    }

    return true;
}

bool FluxEngine::readTimeGuidanceEmbed(int stepIdx, float* outEmbed3072) {
    // Default 4-step shifted-flow timesteps for 512x512 (seq_len = 1024, mu = 2.0306897)
    static const float DEFAULT_TIMESTEPS_1000[4] = {
        1000.0000f,
        958.0967f,
        883.9823f,
        717.5080f
    };
    if (stepIdx < 0) stepIdx = 0;
    if (stepIdx >= 4) stepIdx = 3;
    return computeTimeGuidanceEmbed(DEFAULT_TIMESTEPS_1000[stepIdx], outEmbed3072);
}

bool FluxEngine::eulerStep(float* noisyLatent, const float* velocity, int numElements, float dt) {
    if (!noisyLatent || !velocity || numElements <= 0) return false;

    // FlowMatchEulerDiscreteScheduler update: prev_sample = sample + (sigma_next - sigma_curr) * model_output
    for (int i = 0; i < numElements; ++i) {
        noisyLatent[i] += dt * velocity[i];
    }
    return true;
}

bool FluxEngine::validateFinite(const float* data, size_t count) {
    if (!data) return false;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

float FluxEngine::bf16ToFp32(uint16_t b) {
    uint32_t val = static_cast<uint32_t>(b) << 16;
    float result;
    std::memcpy(&result, &val, sizeof(float));
    return result;
}

float FluxEngine::fp16ToFp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x0001;
    uint32_t exp = (h >> 10) & 0x001f;
    uint32_t frac = h & 0x03ff;

    uint32_t res;
    if (exp == 0) {
        if (frac == 0) {
            res = sign << 31;
        } else {
            exp = 1;
            while ((frac & 0x0400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03ff;
            res = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        res = (sign << 31) | 0x7f800000 | (frac << 13);
    } else {
        res = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    }

    float out;
    std::memcpy(&out, &res, sizeof(float));
    return out;
}

} // namespace flux

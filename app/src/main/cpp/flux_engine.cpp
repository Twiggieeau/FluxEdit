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

bool FluxEngine::patchifyLatent(const float* vaeLatent, float* packedTokens, int height, int width) {
    if (!vaeLatent || !packedTokens || height <= 0 || width <= 0) return false;

    // VAE latent dimensions: [1, 32, H/8, W/8]
    int hLatent = height / 8;
    int wLatent = width / 8;
    int hPatches = hLatent / 2; // (H/16)
    int wPatches = wLatent / 2; // (W/16)
    int numTokens = hPatches * wPatches;

    // Packed tokens shape: [numTokens, 128]
    // 128 = 2 * 2 * 32 (spatial patch of 2x2 with 32 channels)
    for (int py = 0; py < hPatches; ++py) {
        for (int px = 0; px < wPatches; ++px) {
            int tokenIdx = py * wPatches + px;
            float* tokenPtr = packedTokens + tokenIdx * 128;

            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int patchSpatialIdx = dy * 2 + dx;
                    int y = py * 2 + dy;
                    int x = px * 2 + dx;

                    for (int c = 0; c < 32; ++c) {
                        int vaeIdx = c * (hLatent * wLatent) + y * wLatent + x;
                        int packedIdx = patchSpatialIdx * 32 + c;
                        tokenPtr[packedIdx] = vaeLatent[vaeIdx];
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

    for (int py = 0; py < hPatches; ++py) {
        for (int px = 0; px < wPatches; ++px) {
            int tokenIdx = py * wPatches + px;
            const float* tokenPtr = packedTokens + tokenIdx * 128;

            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int patchSpatialIdx = dy * 2 + dx;
                    int y = py * 2 + dy;
                    int x = px * 2 + dx;

                    for (int c = 0; c < 32; ++c) {
                        int vaeIdx = c * (hLatent * wLatent) + y * wLatent + x;
                        int packedIdx = patchSpatialIdx * 32 + c;
                        vaeLatent[vaeIdx] = tokenPtr[packedIdx];
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
    int numTokens = hPatches * wPatches;
    int halfDim = hiddenDim / 2;

    float thetaBase = 10000.0f;
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
    float thetaBase = 10000.0f;
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

bool FluxEngine::readTimeGuidanceEmbed(int stepIdx, float* outEmbed3072) {
    if (!outEmbed3072 || timeGuidanceRawBuffer.empty()) return false;
    if (stepIdx < 0 || stepIdx >= 4) stepIdx = 0;

    // 4 denoising steps, each embedding is 3,072 BF16 values (6,144 bytes per step)
    size_t offset = static_cast<size_t>(stepIdx) * 3072 * 2;
    if (offset + 3072 * 2 > timeGuidanceRawBuffer.size()) {
        LOGE("Offset %zu exceeds time guidance buffer size %zu", offset, timeGuidanceRawBuffer.size());
        return false;
    }

    const uint16_t* bf16Ptr = reinterpret_cast<const uint16_t*>(timeGuidanceRawBuffer.data() + offset);
    for (int i = 0; i < 3072; ++i) {
        outEmbed3072[i] = bf16ToFp32(bf16Ptr[i]);
    }
    return true;
}

bool FluxEngine::eulerStep(float* noisyLatent, const float* velocity, int numElements, float dt) {
    if (!noisyLatent || !velocity || numElements <= 0) return false;

    // Euler update: x_{t+1} = x_t + dt * velocity (direction points towards target image)
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

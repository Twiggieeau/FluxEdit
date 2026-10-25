#ifndef FLUX_ENGINE_H
#define FLUX_ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace flux {

enum class WorkflowFamily {
    TEXT_TO_IMAGE = 0,     // kc_* (0 conditioning images)
    SINGLE_IMAGE_EDIT = 1, // kce_* (1 conditioning image, temporal pos 10)
    TWO_IMAGE_EDIT = 2     // kce2_* (2 conditioning images, temporal pos 10 & 20)
};

struct WeightMapping {
    std::string name;
    int fd;
    void* mmapPtr;
    size_t size;
};

class FluxEngine {
public:
    FluxEngine();
    ~FluxEngine();

    bool initialize(const std::string& modelDir);
    void release();

    // Latent operations according to FLUX.2 Klein contract
    bool generateGaussianNoise(float* outLatent, int numElements, uint64_t seed);
    bool interpolateInitLatent(float* outNoisyLatent, const float* encodedLatent, const float* noise, int numElements, float initStrength);
    bool patchifyLatent(const float* vaeLatent, float* packedTokens, int height, int width);
    bool unpatchifyLatent(const float* packedTokens, float* vaeLatent, int height, int width);

    // Rotary Positional Embedding (RoPE) tensor generators
    bool computeRoPEImage(float* outRotary, int height, int width, int hiddenDim = 128);
    bool computeRoPEText(float* outRotary, int seqLen = 512, int hiddenDim = 128);
    bool computeJointRoPE(float* outCos, float* outSin, int height, int width, int textSeqLen, int numCondImages);

    // Time & Guidance vector computation (3,072 dimensions)
    bool readTimeGuidanceEmbed(int stepIdx, float* outEmbed3072);
    bool computeTimeGuidanceEmbed(float timestep1000, float* outEmbed3072);

    // Flow scheduler Euler step: x_{t+1} = x_t + dt * v
    bool eulerStep(float* noisyLatent, const float* velocity, int numElements, float dt);

    // Tensor validation (reject NaN / Inf)
    static bool validateFinite(const float* data, size_t count);

    // Helper conversion routines
    static float fp16ToFp32(uint16_t h);
    static float bf16ToFp32(uint16_t b);

private:
    std::string modelDirectory;
    std::unordered_map<std::string, WeightMapping> weightSidecars;
    std::vector<uint8_t> timeGuidanceRawBuffer;
    bool isInitialized = false;

    bool loadTimeGuidanceBuffer();
    bool mapWeightSidecars();
};

} // namespace flux

#endif // FLUX_ENGINE_H

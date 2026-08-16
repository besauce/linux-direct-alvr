#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "VkContext.hpp"
#include "utils.hpp"

#include "ffmpeg_helper.h"

namespace alvr::render {

constexpr u32 ImageCount = 6;
constexpr usize StagingImgCount = 2;

enum class HandleType {
    None,
    DmaBuf,
    OpaqueFd,
};

struct Image {
    vk::Image image = VK_NULL_HANDLE;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::DeviceMemory memory = VK_NULL_HANDLE;
    vk::ImageView view = VK_NULL_HANDLE;

    void destroy(VkContext const& ctx) {
        ctx.dev.destroy(view);
        ctx.dev.free(memory);
        ctx.dev.destroy(image);
    }
};

struct Output {
    Image image;
    DrmImage drm;
    // VkSemaphore semaphore;
    VkImageCreateInfo imageCI;
    VkDeviceSize size;
    // Format SteamVR submitted, kept separate from imageCI.format which is
    // normalized to UNORM for storage/FFmpeg/DRM compatibility.
    VkFormat contentFormat;
};

struct PipelineCreateInfo {
    std::vector<u8> shaderData;
    std::vector<vk::SpecializationMapEntry> specs;
    std::vector<u8> specData;
};

// Push-constant payload for the rotational reprojection fold in ffr.comp and
// quad.comp. Field offsets must match the GLSL Rewarp block (0/64/80/96).
// enabled == 0 reproduces the unwarped sampling exactly, so a zero-initialized
// struct is a correct passthrough.
struct WarpParams {
    float rotation[16] = {}; // column-major mat4, rotation in the upper-left 3x3
    float leftTans[4] = {}; // tanLeft, tanRight, tanUp, tanDown
    float rightTans[4] = {};
    u32 enabled = 0;
    u32 _padding[3] = {};
};
static_assert(sizeof(WarpParams) == 112);

struct RendererCreateInfo {
    vk::Format format;
    vk::Extent2D inputEyeExtent;
    vk::Extent2D outputExtent;
    std::array<int, ImageCount> inputImgFds;
};

namespace detail {

    class RenderPipeline {
        vk::ShaderModule shader;
        vk::PipelineLayout pipeLayout;
        vk::Pipeline pipe;

    public:
        RenderPipeline(
            VkContext const& ctx, vk::DescriptorSetLayout& layout, PipelineCreateInfo& pipelineCI
        );

        void render(
            VkContext const& ctx,
            vk::CommandBuffer cmdBuf,
            vk::ImageView in,
            vk::ImageView out,
            vk::Extent2D outSize,
            WarpParams const* warp
        );

        void destroy(VkContext const& ctx);
    };

}

class Renderer {
public:
    // Replace the imported input images with a new swap texture set of the
    // same extent and format. Inputs are only ever copy sources, so nothing
    // else in the pipeline needs rebuilding. Caller must ensure no rendering
    // is in flight (render() waits its completion semaphore before returning).
    void swapInputs(VkContext const& vkCtx, std::array<int, ImageCount> const& fds);

private:
    vk::Extent2D eyeExtent;
    vk::Extent2D outExtent;

    Image inputImages[ImageCount];
    vk::ImageCreateInfo savedInputImgCI {};
    Image stagingImgs[StagingImgCount];
    Output output;

    vk::QueryPool timestampPool;
    vk::CommandPool cmdPool;
    vk::CommandBuffer cmdBuf;
    vk::Sampler sampler;
    vk::DescriptorSetLayout descLayout;

    std::vector<detail::RenderPipeline> pipes;

    WarpParams warpParams {};

    vk::Semaphore renderFinishedSem;
    uint64_t nextSignalValue = 1;
    uint64_t lastSubmittedValue = 0;

public:
    vk::Semaphore getRenderFinishedSemaphore() const { return renderFinishedSem; }
    uint64_t getLastSubmittedValue() const { return lastSubmittedValue; }
    void waitForRenderDone(VkContext const& vkCtx) const {
        if (renderFinishedSem == VK_NULL_HANDLE) {
            return;
        }

        vk::SemaphoreWaitInfo waitInfo {
            .semaphoreCount = 1,
            .pSemaphores = &renderFinishedSem,
            .pValues = &lastSubmittedValue,
        };

        auto result = vkCtx.dev.waitSemaphores(waitInfo, UINT64_MAX);
        if (result != vk::Result::eSuccess && result != vk::Result::eTimeout) {
            throw std::runtime_error("Failed waiting for render completion semaphore");
        }
    }

    Renderer(
        VkContext const& vkCtx,
        RendererCreateInfo& createInfo,
        std::vector<PipelineCreateInfo> pipeCIs
    );

    // TODO: Import output (somehow?)

    // NOTE: Use the output immediately afterwards, as this synchronizes to the end of gpu
    // operations
    // waitFds, when given, are sync_file fds imported as temporary semaphores
    // the submission waits on before its transfer stage (the input writers'
    // fences). Successful import hands the fd to Vulkan; failed import closes
    // it here.
    void render(VkContext& vkCtx, u32 leftIdx, u32 rightIdx, int const waitFds[2] = nullptr);

    // Warp parameters pushed to every pass of the next render() call. The
    // caller owns the cadence: set before each render, or leave the default
    // zero struct for exact passthrough.
    void setWarpParams(WarpParams const& params) { warpParams = params; }

    Output getOutput() { return output; }

    void destroy(VkContext const& ctx);
};

}

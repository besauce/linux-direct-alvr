#include "ffmpeg_helper.h"
#include "VkContext.hpp"

#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include "alvr_server/Logger.h"
#include "alvr_server/bindings.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavutil/avutil.h>
}

namespace {

// TODO: I don't think this selects the optimal format actually
// Seeing as we don't want any re-encoding here yet
// We just want it to make the drm image 1:1
AVPixelFormat vk_format_to_av_format(vk::Format vk_fmt) {
    for (int f = AV_PIX_FMT_NONE; f < AV_PIX_FMT_NB; ++f) {
        auto current_fmt = av_vkfmt_from_pixfmt(AVPixelFormat(f));
        if (current_fmt and *current_fmt == (VkFormat)vk_fmt)
            return AVPixelFormat(f);
    }

    throw std::runtime_error("unsupported vulkan pixel format " + std::to_string((VkFormat)vk_fmt));
}
}

std::string alvr::AvException::makemsg(const std::string& msg, int averror) {
    char av_msg[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(averror, av_msg, sizeof(av_msg));
    return msg + " " + av_msg;
}

// alvr::VkFrameCtx::VkFrameCtx(VkContext & vkContext, vk::ImageCreateInfo image_create_info)
// {
//   AVHWFramesContext *frames_ctx = NULL;
//   int err = 0;

//   if (!(ctx = av_hwframe_ctx_alloc(vkContext.ctx))) {
//     throw std::runtime_error("Failed to create vulkan frame context.");
//   }
//   frames_ctx = (AVHWFramesContext *)(ctx->data);
//   frames_ctx->format = AV_PIX_FMT_VULKAN;
//   frames_ctx->sw_format = vk_format_to_av_format(image_create_info.format);
//   frames_ctx->width = image_create_info.extent.width;
//   frames_ctx->height = image_create_info.extent.height;
//   frames_ctx->initial_pool_size = 0;
//   if ((err = av_hwframe_ctx_init(ctx)) < 0) {
//     av_buffer_unref(&ctx);
//     throw alvr::AvException("Failed to initialize vulkan frame context:", err);
//   }
// }

// alvr::VkFrameCtx::~VkFrameCtx()
// {
//   av_buffer_unref(&ctx);
// }

alvr::VkFrameCtx::~VkFrameCtx() { av_buffer_unref(&ctx); }

// The colorspace of a frame is keyed off the format SteamVR submitted, not
// the created image format, which is normalized to UNORM before it gets here.
// Static: these are file-local details of the mapping, not API.
static AVColorPrimaries vk_format_to_av_primaries(VkFormat vk_fmt) {
    switch (vk_fmt) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return AVCOL_PRI_BT709;
    default:
        return AVCOL_PRI_UNSPECIFIED;
    }
}

static AVColorTransferCharacteristic vk_format_to_av_trc(VkFormat vk_fmt) {
    switch (vk_fmt) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return AVCOL_TRC_IEC61966_2_1;
    default:
        return AVCOL_TRC_UNSPECIFIED;
    }
}

static AVColorSpace vk_format_to_av_colorspace(VkFormat vk_fmt) {
    switch (vk_fmt) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return AVCOL_SPC_BT709;
    default:
        return AVCOL_SPC_UNSPECIFIED;
    }
}

static AVColorRange vk_format_to_av_range(VkFormat vk_fmt) {
    switch (vk_fmt) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
        // The submitted images are RGB, which is full range by definition.
        // Range conversion for the encoder happens downstream.
        return AVCOL_RANGE_JPEG;
    default:
        return AVCOL_RANGE_UNSPECIFIED;
    }
}

alvr::VkFrame::VkFrame(
    const VkContext& vk_ctx,
    VkImage image,
    VkImageCreateInfo image_info,
    VkDeviceSize size,
    VkDeviceMemory memory,
    DrmImage drm,
    VkFormat content_format
)
    : vkimage(image)
    , vkimageinfo(image_info)
    , contentformat(content_format) {
    device = vk_ctx.dev;
    avformat = vk_format_to_av_format(vk::Format(image_info.format));

    av_drmframe = (AVDRMFrameDescriptor*)malloc(sizeof(AVDRMFrameDescriptor));
    av_drmframe->nb_objects = 1;
    av_drmframe->objects[0].fd = drm.fd;
    av_drmframe->objects[0].size = size;
    av_drmframe->objects[0].format_modifier = drm.modifier;
    av_drmframe->nb_layers = 1;
    av_drmframe->layers[0].format = drm.format;
    av_drmframe->layers[0].nb_planes = drm.planes;
    for (uint32_t i = 0; i < drm.planes; ++i) {
        av_drmframe->layers[0].planes[i].object_index = 0;
        av_drmframe->layers[0].planes[i].pitch = drm.strides[i];
        av_drmframe->layers[0].planes[i].offset = drm.offsets[i];
    }

    av_vkframe = av_vk_frame_alloc();
    av_vkframe->img[0] = image;
    av_vkframe->tiling = image_info.tiling;
    av_vkframe->mem[0] = memory;
    av_vkframe->size[0] = size;
    av_vkframe->layout[0] = VK_IMAGE_LAYOUT_UNDEFINED;

    VkExportSemaphoreCreateInfo exportInfo = {};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreTypeCreateInfo timelineInfo = {};
    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.pNext = &exportInfo;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;

    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &timelineInfo;
    vkCreateSemaphore(device, &semInfo, nullptr, &av_vkframe->sem[0]);
}

AVColorPrimaries alvr::VkFrame::colorPrimaries() const {
    return vk_format_to_av_primaries(contentformat);
}

AVColorTransferCharacteristic alvr::VkFrame::colorTransfer() const {
    return vk_format_to_av_trc(contentformat);
}

AVColorSpace alvr::VkFrame::colorSpace() const {
    return vk_format_to_av_colorspace(contentformat);
}

AVColorRange alvr::VkFrame::colorRange() const {
    return vk_format_to_av_range(contentformat);
}

alvr::VkFrame::~VkFrame() {
    free(av_drmframe);
    if (av_vkframe) {
        vkDestroySemaphore(device, av_vkframe->sem[0], nullptr);
        av_free(av_vkframe);
    }
}

std::unique_ptr<AVFrame, std::function<void(AVFrame*)>>
alvr::VkFrame::make_av_frame(VkFrameCtx& frame_ctx) {
    std::unique_ptr<AVFrame, std::function<void(AVFrame*)>> frame {
        av_frame_alloc(), [](AVFrame* p) { av_frame_free(&p); }
    };
    frame->width = vkimageinfo.extent.width;
    frame->height = vkimageinfo.extent.height;
    frame->hw_frames_ctx = av_buffer_ref(frame_ctx.ctx);
    frame->data[0] = (uint8_t*)av_vkframe;
    frame->format = AV_PIX_FMT_VULKAN;
    frame->buf[0] = av_buffer_alloc(1);
    frame->color_primaries = colorPrimaries();
    frame->color_trc = colorTransfer();
    frame->colorspace = colorSpace();
    frame->color_range = colorRange();
    frame->pts = std::chrono::steady_clock::now().time_since_epoch().count();

    return frame;
}

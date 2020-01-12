#include "vulkan_interpolator/VulkanInterpolatorHeadless.hpp"

#include "shaders.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

#include <delaunator.hpp>

#define DEBUG

#ifdef DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  (void)pUserData;
  std::cerr << "[vk::Validation]["
            << vk::to_string(
                   (vk::DebugUtilsMessageSeverityFlagBitsEXT)messageSeverity)
            << "|"
            << vk::to_string((vk::DebugUtilsMessageTypeFlagBitsEXT)messageType)
            << "]: " << pCallbackData->pMessage << std::endl;
  if ((vk::DebugUtilsMessageSeverityFlagBitsEXT)messageSeverity ==
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
    throw std::runtime_error("Vulkan error");
  return VK_FALSE;
}

std::unordered_map<VkInstance, PFN_vkCreateDebugUtilsMessengerEXT>
    CreateDebugUtilsMessengerEXTDispatchTable;
std::unordered_map<VkInstance, PFN_vkDestroyDebugUtilsMessengerEXT>
    DestroyDebugUtilsMessengerEXTDispatchTable;
std::unordered_map<VkInstance, PFN_vkSubmitDebugUtilsMessageEXT>
    SubmitDebugUtilsMessageEXTDispatchTable;

void loadDebugUtilsCommands(VkInstance instance) {
  PFN_vkVoidFunction temp_fp;

  temp_fp = vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (!temp_fp)
    throw "Failed to load vkCreateDebugUtilsMessengerEXT"; // check shouldn't
                                                           // be necessary
                                                           // (based on spec)
  CreateDebugUtilsMessengerEXTDispatchTable[instance] =
      reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(temp_fp);

  temp_fp = vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (!temp_fp)
    throw "Failed to load vkDestroyDebugUtilsMessengerEXT"; // check shouldn't
                                                            // be necessary
                                                            // (based on spec)
  DestroyDebugUtilsMessengerEXTDispatchTable[instance] =
      reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(temp_fp);

  temp_fp = vkGetInstanceProcAddr(instance, "vkSubmitDebugUtilsMessageEXT");
  if (!temp_fp)
    throw "Failed to load vkSubmitDebugUtilsMessageEXT"; // check shouldn't be
                                                         // necessary (based on
                                                         // spec)
  SubmitDebugUtilsMessageEXTDispatchTable[instance] =
      reinterpret_cast<PFN_vkSubmitDebugUtilsMessageEXT>(temp_fp);
}

void unloadDebugUtilsCommands(VkInstance instance) {
  CreateDebugUtilsMessengerEXTDispatchTable.erase(instance);
  DestroyDebugUtilsMessengerEXTDispatchTable.erase(instance);
  SubmitDebugUtilsMessageEXTDispatchTable.erase(instance);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pMessenger) {
  auto dispatched_cmd = CreateDebugUtilsMessengerEXTDispatchTable.at(instance);
  return dispatched_cmd(instance, pCreateInfo, pAllocator, pMessenger);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(
    VkInstance instance, VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks *pAllocator) {
  auto dispatched_cmd = DestroyDebugUtilsMessengerEXTDispatchTable.at(instance);
  return dispatched_cmd(instance, messenger, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkSubmitDebugUtilsMessageEXT(
    VkInstance instance, VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData) {
  auto dispatched_cmd = SubmitDebugUtilsMessageEXTDispatchTable.at(instance);
  return dispatched_cmd(instance, messageSeverity, messageTypes, pCallbackData);
}
#endif
namespace {
auto kernel2string(const std::string_view &view) {
  std::ostringstream kernel;
  for (auto &c : view)
    kernel << std::hex << std::setfill('0') << std::setw(2) << (uint32_t)c;
  return kernel.str();
}

} // namespace

namespace vulkan_interpolator {

void HeadlessInterpolator::interpolate(const int nPoints, const float *points,
                                       const float *values, const int width,
                                       const int height, const int stride_bytes,
                                       float *output, float dt, float db,
                                       float dl, float dr, float sx, float sy) {
  std::vector<int> indicies;
  std::cout << "Npoints: " << nPoints << std::endl;
  PrepareInterpolation(nPoints, points, indicies);
  const int nTri = indicies.size() / 3;
  interpolate(nPoints, points, values, nTri, indicies.data(), width, height,
              stride_bytes, output, dt, db, dl, dr, sx, sy);
}

void HeadlessInterpolator::PrepareInterpolation(const int nPoints,
                                                const float *points,
                                                std::vector<int> &indicies) {
  std::vector<double> pts(points, points + nPoints * 2);
  delaunator::Delaunator d(pts);
  indicies.clear();
  indicies.reserve(d.triangles.size());
  for (auto &id : d.triangles)
    indicies.emplace_back(id);
}

void HeadlessInterpolator::interpolate(
    const int nPoints, const float *points_, const float *values,
    const int nTriangles, const int *indicies_, const int width_,
    const int height_, const int stride_bytes, float *output, float dt,
    float db, float dl, float dr, float sx, float sy) {
  std::lock_guard<std::mutex> lock(mutex);
  points = nPoints;
  height = height_;
  width = width_;
  indicies = nTriangles * 3;
  std::cout << "Triangles: " << nTriangles << std::endl;
  // so what is worse -- reallocating output buffer vs copying larger size?!
  // should not matter in "real life" (= multiple interpolations of the same
  // image size)
  setupCopyImage();

  setupVertices();

  float scale_w = 2.f / (width - 1 + dr - dl) * sx,
        scale_h = 2.f / (height - 1 + db - dt) * sy;
  float bias_w = -(width - 1 + dl + dr) * scale_w / 2.f;
  float bias_h = -(height - 1 + dt + db) * scale_h / 2.f;

  std::cout << heightAllocated << " x " << widthAllocated << " " << scale_w
            << " " << scale_h << " " << bias_w << " " << bias_h << std::endl;

  int argout_pts = 0;
  for (int i = 0; i < nPoints; ++i) {
    points_ptr[argout_pts++] = points_[i * 2] * scale_w + bias_w;
    points_ptr[argout_pts++] = points_[i * 2 + 1] * scale_h + bias_h;
    points_ptr[argout_pts++] = values[i];
  }
  memcpy((void *)indicies_ptr, (void *)indicies_, sizeof(int32_t) * indicies);
  device->unmapMemory(*stagingBuffer.first);

  rasterize();
  vk::SubresourceLayout layout = device->getImageSubresourceLayout(
      *outputImage, {vk::ImageAspectFlagBits::eColor});
  const char *data = (const char *)device->mapMemory(
      *outputMem, layout.offset, layout.arrayPitch, vk::MemoryMapFlags{});
  const char *strided_output = reinterpret_cast<const char *>(output);
  for (uint32_t i = 0; i < height; ++i) {
    memcpy((void *)strided_output, (void *)data, width * sizeof(float));
    strided_output += stride_bytes;
    data += layout.rowPitch;
  }
  device->unmapMemory(*outputMem);
}

bool HeadlessInterpolator::allocatePoints() { return points > pointsAllocated; }

bool HeadlessInterpolator::allocateIndicies() {
  return indicies > indiciesAllocated;
}

void HeadlessInterpolator::setupVertices() {
  const size_t mem_points = points * sizeof(float) * 3;
  const size_t mem_indicies = indicies * sizeof(int32_t);
  const size_t mem_total = mem_points + mem_indicies;

  // setup vertex/index buffers
  if (pointsAllocated < points) {
    std::cout << "Allocating vertex buffer: " << pointsAllocated << " -> "
              << points << std::endl;
    vertexBuffer = createBuffer(mem_points,
                                vk::BufferUsageFlagBits::eVertexBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst,
                                vk::MemoryPropertyFlagBits::eDeviceLocal);

    pointsAllocated = points;
  }
  if (indiciesAllocated < indicies) {
    std::cout << "Allocating index buffer: " << indiciesAllocated << " -> "
              << indicies << std::endl;
    indexBuffer = createBuffer(mem_indicies,
                               vk::BufferUsageFlagBits::eIndexBuffer |
                                   vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);
    indiciesAllocated = indicies;
  }

  if (mem_total > stagingAllocated) {
    // allocate staging
    std::cout << "Allocating staging buffer: " << stagingAllocated << " -> "
              << mem_total << std::endl;
    pointsAllocated = points;
    indiciesAllocated = indicies;
    stagingAllocated = mem_total;

    vk::DeviceSize size = stagingAllocated;
    stagingBuffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eHostVisible |
                                     vk::MemoryPropertyFlagBits::eHostCoherent);
  }
  device->mapMemory(*stagingBuffer.first, 0, mem_total, vk::MemoryMapFlags{},
                    &stagingData);
  points_ptr = static_cast<float *>(stagingData);
  // Setup pointers
  indicies_ptr = reinterpret_cast<int32_t *>(static_cast<char *>(stagingData) +
                                             mem_points);

  // setup pointers & copy buffer
  vk::CommandBufferBeginInfo cbegin{};
  copyBuffer->begin(&cbegin);
  vk::BufferCopy region;
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = mem_points;
  copyBuffer->copyBuffer(*stagingBuffer.second, *vertexBuffer.second, 1,
                         &region);
  region.srcOffset = mem_points;
  region.size = mem_indicies;
  copyBuffer->copyBuffer(*stagingBuffer.second, *indexBuffer.second, 1,
                         &region);
  copyBuffer->end();

  // setup render pass
  auto extent = vk::Extent2D{width, height};
  auto beginInfo = vk::CommandBufferBeginInfo{};
  renderBuffer->begin(beginInfo);
  auto renderPassBeginInfo = vk::RenderPassBeginInfo{
      *renderPass, *framebuffer, vk::Rect2D{{0, 0}, extent}, 1, &clearValues};

  vk::DeviceSize offset = 0;
  renderBuffer->bindVertexBuffers(0, 1, &*vertexBuffer.second, &offset);
  renderBuffer->bindIndexBuffer(*indexBuffer.second, 0, vk::IndexType::eUint32);

  renderBuffer->beginRenderPass(renderPassBeginInfo,
                                vk::SubpassContents::eInline);
  renderBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

  std::cout << "Indicies: " << indicies << std::endl;
  renderBuffer->drawIndexed(indicies, 1, 0, 0, 0);
  renderBuffer->endRenderPass();
  renderBuffer->end();
}

bool HeadlessInterpolator::allocateImage() {
  // Should we reallocate iff area > old area?
  return width > widthAllocated || height > heightAllocated;
}

void HeadlessInterpolator::setupCopyImage() {
  vk::ImageCreateInfo imageInfo;
  imageInfo.imageType = vk::ImageType::e2D;
  imageInfo.format = format1d;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  if (allocateImage()) {
    {
      std::cout << "Allocating image: " << heightAllocated << " x "
                << widthAllocated << "  ->  " << height << " x " << width
                << std::endl;
      heightAllocated = std::max(heightAllocated, height);
      widthAllocated = std::max(widthAllocated, width);
      imageInfo.extent.width = widthAllocated;
      imageInfo.extent.height = heightAllocated;
      // color attachment
      imageInfo.usage = vk::ImageUsageFlagBits::eColorAttachment |
                        vk::ImageUsageFlagBits::eTransferSrc;
      image = device->createImageUnique(imageInfo);

      vk::MemoryRequirements reqs = device->getImageMemoryRequirements(*image);
      vk::MemoryAllocateInfo outAllocInfo;
      outAllocInfo.allocationSize = reqs.size;
      outAllocInfo.memoryTypeIndex = findMemoryType(
          reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
      renderMem = device->allocateMemoryUnique(outAllocInfo);
      device->bindImageMemory(*image, *renderMem, 0);

      vk::ImageViewCreateInfo viewInfo;
      viewInfo.viewType = vk::ImageViewType::e2D;
      viewInfo.format = format1d;
      viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
      viewInfo.subresourceRange.baseMipLevel = 0;
      viewInfo.subresourceRange.levelCount = 1;
      viewInfo.subresourceRange.baseArrayLayer = 0;
      viewInfo.subresourceRange.layerCount = 1;
      viewInfo.image = *image;
      imageView = device->createImageViewUnique(viewInfo);
      // TODO: should we create framebuffer if width*height < allocated ?! Or
      // "extent+scissors" thingies are just enough?!
    }
    {
      // copy dst
      imageInfo.initialLayout = vk::ImageLayout::eUndefined;
      imageInfo.tiling = vk::ImageTiling::eLinear;
      imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst;

      outputImage = device->createImageUnique(imageInfo);
      vk::MemoryRequirements reqs =
          device->getImageMemoryRequirements(*outputImage);
      vk::MemoryAllocateInfo outAllocInfo;
      outAllocInfo.allocationSize = reqs.size;
      outAllocInfo.memoryTypeIndex = findMemoryType(
          reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent);
      outputMem = device->allocateMemoryUnique(outAllocInfo);
      device->bindImageMemory(*outputImage, *outputMem, 0);
    }
  }

  if (width == widthLast && height == heightLast)
    return;
  widthLast = width;
  heightLast = height;
  // setup copy back (only transfer subset)
  vk::CommandBufferBeginInfo cbegin{};
  copyBackBuffer->begin(cbegin);

  vk::ImageMemoryBarrier barrier;
  barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;

  // Color -> Src
  barrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
  barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
  barrier.image = *image;
  barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead |
                          vk::AccessFlagBits::eColorAttachmentWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
  copyBackBuffer->pipelineBarrier(
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, nullptr,
      nullptr, barrier);
  // Output -> Dst
  barrier.oldLayout = vk::ImageLayout::eUndefined;
  barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.image = *outputImage;
  barrier.srcAccessMask = {};
  barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
  copyBackBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::PipelineStageFlagBits::eTransfer,
                                  vk::DependencyFlags{}, nullptr, nullptr,
                                  barrier);
  vk::ImageCopy copy;
  copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
  copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
  copy.extent.depth = 1;
  copy.extent.width = width;
  copy.extent.height = height;
  copyBackBuffer->copyImage(*image, vk::ImageLayout::eTransferSrcOptimal,
                            *outputImage, vk::ImageLayout::eTransferDstOptimal,
                            copy);

  // Output -> General
  barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.newLayout = vk::ImageLayout::eGeneral;
  barrier.image = *outputImage;
  barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = {};
  copyBackBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eBottomOfPipe,
                                  vk::DependencyFlags{}, nullptr, nullptr,
                                  barrier);

  copyBackBuffer->end();

  // pipeline setup
  auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo{
      {}, vk::ShaderStageFlagBits::eVertex, *vertexShaderModule, "main"};

  auto fragShaderStageInfo = vk::PipelineShaderStageCreateInfo{
      {}, vk::ShaderStageFlagBits::eFragment, *fragmentShaderModule, "main"};

  auto pipelineShaderStages = std::vector<vk::PipelineShaderStageCreateInfo>{
      vertShaderStageInfo, fragShaderStageInfo};

  vk::VertexInputBindingDescription vertexBinding;
  vertexBinding.binding = 0;
  vertexBinding.stride = sizeof(float) * 3;
  vertexBinding.inputRate = vk::VertexInputRate::eVertex;

  vk::VertexInputAttributeDescription vertexAttributes[] = {
      {0, 0, format2d, 0}, {1, 0, format1d, 2 * sizeof(float)}};

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
  vertexInputInfo.pVertexAttributeDescriptions = vertexAttributes;
  vertexInputInfo.pVertexBindingDescriptions = &vertexBinding;
  vertexInputInfo.vertexAttributeDescriptionCount = 2;
  vertexInputInfo.vertexBindingDescriptionCount = 1;

  auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo{
      {}, vk::PrimitiveTopology::eTriangleList, false};

  auto extent = vk::Extent2D{width, height};

  auto viewport = vk::Viewport{
      0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height),
      0.0f, 1.0f};

  auto scissor = vk::Rect2D{{0, 0}, extent};

  auto viewportState =
      vk::PipelineViewportStateCreateInfo{{}, 1, &viewport, 1, &scissor};

  auto rasterizer = vk::PipelineRasterizationStateCreateInfo{
      {},
      /*depthClamp*/ false,
      /*rasterizeDiscard*/ false,
      vk::PolygonMode::eFill,
      {},
      /*frontFace*/ vk::FrontFace::eClockwise,
      {},
      {},
      {},
      {},
      1.0f};

  auto multisampling = vk::PipelineMultisampleStateCreateInfo{
      {}, vk::SampleCountFlagBits::e1, false, 1.0};

  auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState{
      {},
      /*srcCol*/ vk::BlendFactor::eOne,
      /*dstCol*/ vk::BlendFactor::eZero,
      /*colBlend*/ vk::BlendOp::eAdd,
      /*srcAlpha*/ vk::BlendFactor::eOne,
      /*dstAlpha*/ vk::BlendFactor::eZero,
      /*alphaBlend*/ vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

  auto colorBlending = vk::PipelineColorBlendStateCreateInfo{
      {},
      /*logicOpEnable=*/false,
      vk::LogicOp::eCopy,
      /*attachmentCount=*/1,
      /*colourAttachments=*/&colorBlendAttachment};

  pipelineLayout = device->createPipelineLayoutUnique({}, nullptr);

  auto colorAttachment =
      vk::AttachmentDescription{{},
                                format1d,
                                vk::SampleCountFlagBits::e1,
                                vk::AttachmentLoadOp::eClear,
                                vk::AttachmentStoreOp::eStore,
                                {},
                                {},
                                {},
                                vk::ImageLayout::eColorAttachmentOptimal};

  auto colourAttachmentRef =
      vk::AttachmentReference{0, vk::ImageLayout::eColorAttachmentOptimal};

  auto subpass = vk::SubpassDescription{{},
                                        vk::PipelineBindPoint::eGraphics,
                                        /*inAttachmentCount*/ 0,
                                        nullptr,
                                        1,
                                        &colourAttachmentRef};

  std::array<vk::SubpassDependency, 1> subpassDependency = {
      vk::SubpassDependency{VK_SUBPASS_EXTERNAL, 0,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::AccessFlagBits::eColorAttachmentRead |
                                vk::AccessFlagBits::eColorAttachmentWrite,
                            vk::DependencyFlagBits::eByRegion},
  };

  renderPass = device->createRenderPassUnique(vk::RenderPassCreateInfo{
      {}, 1, &colorAttachment, 1, &subpass, 1, &subpassDependency[0]});
  framebuffer = device->createFramebufferUnique(vk::FramebufferCreateInfo{
      {}, *renderPass, 1, &(*imageView), widthAllocated, heightAllocated, 1});

  auto pipelineCreateInfo =
      vk::GraphicsPipelineCreateInfo{{},
                                     2,
                                     pipelineShaderStages.data(),
                                     &vertexInputInfo,
                                     &inputAssembly,
                                     nullptr,
                                     &viewportState,
                                     &rasterizer,
                                     &multisampling,
                                     nullptr,
                                     &colorBlending,
                                     nullptr,
                                     *pipelineLayout,
                                     *renderPass,
                                     0};

  pipeline = device->createGraphicsPipelineUnique({}, pipelineCreateInfo);
}

uint32_t HeadlessInterpolator::findMemoryType(
    uint32_t mask, const vk::MemoryPropertyFlags &properties) {
  auto props = physicalDevice.getMemoryProperties();
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((mask & (1 << i)) &&
        (props.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  }
  throw std::runtime_error("Incompatible memory type");
}

HeadlessInterpolator::BufferMem
HeadlessInterpolator::createBuffer(size_t sz, const vk::BufferUsageFlags &usage,
                                   const vk::MemoryPropertyFlags &flags) {
  vk::BufferCreateInfo bufferInfo;
  bufferInfo.size = sz;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;

  vk::UniqueBuffer buffer = device->createBufferUnique(bufferInfo);

  vk::MemoryRequirements reqs = device->getBufferMemoryRequirements(*buffer);

  vk::MemoryAllocateInfo allocInfo;
  allocInfo.allocationSize = reqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(reqs.memoryTypeBits, flags);

  vk::UniqueDeviceMemory mem = device->allocateMemoryUnique(allocInfo);
  device->bindBufferMemory(*buffer, *mem, 0);
  return std::make_pair(std::move(mem), std::move(buffer));
}

HeadlessInterpolator::~HeadlessInterpolator() {
  //  device->unmapMemory(*stagingBuffer.first);
}

HeadlessInterpolator::HeadlessInterpolator(
    const std::vector<size_t> &allowed_devices,
    const InterpolationOptions &opts) {
  std::cout << "Vertex shader: (" << shaders::shader_vert_spv.size()
            << " bytes)" << kernel2string(shaders::shader_vert_spv)
            << std::endl;
  std::cout << "Fragment shader: (" << shaders::shader_frag_spv.size()
            << " bytes)" << kernel2string(shaders::shader_frag_spv)
            << std::endl;
  vk::ApplicationInfo appInfo("Hello Triangle", VK_MAKE_VERSION(1, 0, 0),
                              "No Engine", VK_MAKE_VERSION(1, 0, 0),
                              VK_API_VERSION_1_0);

  std::vector<const char *> glfwExtensionsVector;
#ifdef DEBUG
  glfwExtensionsVector.push_back("VK_EXT_debug_utils");
#endif
  auto layers = std::vector<const char *>{
#ifdef DEBUG
      "VK_LAYER_LUNARG_standard_validation"
#endif
  };

  height = opts.heightPreallocated;
  width = opts.widthPreallocated;
  points = opts.pointsPreallocated;
  indicies = opts.indiciesPreallocated;

  instance = vk::createInstanceUnique(
      vk::InstanceCreateInfo{{},
                             &appInfo,
                             static_cast<uint32_t>(layers.size()),
                             layers.data(),
                             static_cast<uint32_t>(glfwExtensionsVector.size()),
                             glfwExtensionsVector.data()});
  auto physicalDevices = instance->enumeratePhysicalDevices();
  std::cout << "#Devices = " << physicalDevices.size() << std::endl;

  physicalDevice = physicalDevices[allowed_devices[0]];
#ifdef DEBUG
  loadDebugUtilsCommands(*instance);

  messenger = instance->createDebugUtilsMessengerEXTUnique(
      vk::DebugUtilsMessengerCreateInfoEXT{
          {},
          vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
              vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
              vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
              vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
          vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
              vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
              vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
          debugCallback},
      nullptr);
#endif
  auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

  size_t graphicsQueueFamilyIndex = std::distance(
      queueFamilyProperties.begin(),
      std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                   [](vk::QueueFamilyProperties const &qfp) {
                     return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
                   }));

  std::vector<uint32_t> FamilyIndices = {(uint32_t)graphicsQueueFamilyIndex};

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

  float queuePriority = 0.0f;
  for (int queueFamilyIndex : FamilyIndices) {
    queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{
        vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(queueFamilyIndex),
        1, &queuePriority});
  }

  device = physicalDevice.createDeviceUnique(
      vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queueCreateInfos.size(),
                           queueCreateInfos.data(), 0, nullptr, 0, nullptr));

  auto vertShaderCreateInfo = vk::ShaderModuleCreateInfo{
      {},
      shaders::shader_vert_spv.size(),
      reinterpret_cast<const uint32_t *>(shaders::shader_vert_spv.data())};
  vertexShaderModule = device->createShaderModuleUnique(vertShaderCreateInfo);
  auto fragShaderCreateInfo = vk::ShaderModuleCreateInfo{
      {},
      shaders::shader_frag_spv.size(),
      reinterpret_cast<const uint32_t *>(shaders::shader_frag_spv.data())};
  fragmentShaderModule = device->createShaderModuleUnique(fragShaderCreateInfo);

  commandPoolUnique = device->createCommandPoolUnique(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       static_cast<uint32_t>(graphicsQueueFamilyIndex)});
#if 0
  createVertexBuffer();
  createIndexBuffer();
  createStagingBuffer();
#endif

  auto bar = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
      *commandPoolUnique, vk::CommandBufferLevel::ePrimary, 3));
  renderBuffer = std::move(bar[0]);
  copyBuffer = std::move(bar[1]);
  copyBackBuffer = std::move(bar[2]);

  deviceQueue = device->getQueue(graphicsQueueFamilyIndex, 0);

  for (int i = 0; i < 4; ++i)
    clearValues.color.float32[i] = std::nan("");

  vk::FenceCreateInfo info;
  fence = device->createFenceUnique(info);

  setupCopyImage();
  setupVertices();
  device->unmapMemory(*stagingBuffer.first);
}

void HeadlessInterpolator::rasterize() {
  vk::CommandBuffer buffers[] = {*copyBuffer, *renderBuffer, *copyBackBuffer};
  vk::SubmitInfo submitInfo;
  submitInfo.waitSemaphoreCount = submitInfo.signalSemaphoreCount = 0;
  submitInfo.pWaitDstStageMask = nullptr;
  submitInfo.commandBufferCount = 3;
  submitInfo.pCommandBuffers = buffers;

  device->resetFences(1, &*fence);
  deviceQueue.submit(submitInfo, fence.get());
  device->waitForFences(1, &*fence, true, std::numeric_limits<uint64_t>::max());
}

} // namespace vulkan_interpolator

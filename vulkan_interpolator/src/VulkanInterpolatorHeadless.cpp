#include "vulkan_interpolator/VulkanInterpolatorHeadless.hpp"

#include "shaders.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

#include <delaunator.hpp>

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
    exit(-1);
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

void HeadlessInterpolator::createPoints() {
  std::uniform_real_distribution<float> u01(0, 1), u11(-1, 1);
  points.resize(N_PTS * 3);
  int idx = 0;
  std::vector<double> coords;
  for (auto &p : points) {

    p = (idx % 3) < 2 ? u11(rng) : u01(rng);
    if (idx % 3 == 1) {
      coords.push_back(points[idx - 1]);
      coords.push_back(points[idx]);
    }
    ++idx;
  }

  auto dstart = std::chrono::high_resolution_clock::now();
  std::cout << "Delaunator: " << std::endl;
  delaunator::Delaunator d(coords);
  if (d.triangles.size() > IDX_MAX_CNT) {
    std::cout << "FUCKED UP!" << std::endl;
    exit(-1);
  }
  nTris = d.triangles.size() / 3;
  std::cout << "Triangles: " << nTris << std::endl;

  indicies.resize(nTris * 3);
  for (int i = 0; i < nTris; ++i) {
    for (int j = 0; j < 3; ++j) {
      indicies[i * 3 + j] = d.triangles[i * 3 + j];
    }
  }
  stageData();
  auto dstop = std::chrono::high_resolution_clock::now();
  std::cout << "D: " << (1e9 / (dstop - dstart).count()) << "FPS" << std::endl;
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
  return std::make_pair(std::move(buffer), std::move(mem));
}
void HeadlessInterpolator::createStagingBuffer() {
  vk::DeviceSize size = PTS_SIZE + IDX_MAX_SIZE;
  stagingBuffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                               vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent);

  vk::CommandBufferAllocateInfo allocInfo;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandPool = *commandPoolUnique;
  allocInfo.commandBufferCount = 1;

  auto foo = device->allocateCommandBuffersUnique(allocInfo);
  copyBuffer = std::move(foo[0]);

  vk::CommandBufferBeginInfo cbegin{};
  copyBuffer->begin(&cbegin);
  vk::BufferCopy regions[2];
  regions[1].srcOffset = PTS_SIZE;
  regions[0].srcOffset = 0;
  regions[0].dstOffset = regions[1].dstOffset = 0;
  regions[0].size = PTS_SIZE;
  regions[1].size = IDX_MAX_SIZE;
  copyBuffer->copyBuffer(*stagingBuffer.first, *vertexBuffer.first, 1, regions);
  copyBuffer->copyBuffer(*stagingBuffer.first, *indexBuffer.first, 1,
                         regions + 1);
  copyBuffer->end();
}

void HeadlessInterpolator::stageData() {
  int IDX_SIZE = nTris * 3 * sizeof(int);
  device->mapMemory(*stagingBuffer.second, 0, PTS_SIZE + IDX_MAX_SIZE,
                    vk::MemoryMapFlags{}, &stagingData);
  { memcpy(stagingData, points.data(), N_PTS * 3 * sizeof(float)); }
  { memcpy((uint8_t *)stagingData + PTS_SIZE, indicies.data(), IDX_SIZE); }
  device->unmapMemory(*stagingBuffer.second);
}

void HeadlessInterpolator::createVertexBuffer() {
  vertexBuffer = createBuffer(PTS_SIZE,
                              vk::BufferUsageFlagBits::eVertexBuffer |
                                  vk::BufferUsageFlagBits::eTransferDst,
                              vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void HeadlessInterpolator::createIndexBuffer() {
  indexBuffer = createBuffer(IDX_MAX_SIZE,
                             vk::BufferUsageFlagBits::eIndexBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst,
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
}

HeadlessInterpolator::~HeadlessInterpolator() {
  //  device->unmapMemory(*stagingBuffer.second);
}

HeadlessInterpolator::HeadlessInterpolator(
    const std::vector<size_t> &allowed_devices) {

  (void)allowed_devices;
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

  auto extent = vk::Extent2D{width, height};

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
      {0, 0, format2d, 0},
      {1, 0, format1d, 2 * sizeof(float)}};

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
  vertexInputInfo.pVertexAttributeDescriptions = vertexAttributes;
  vertexInputInfo.pVertexBindingDescriptions = &vertexBinding;
  vertexInputInfo.vertexAttributeDescriptionCount = 2;
  vertexInputInfo.vertexBindingDescriptionCount = 1;

  auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo{
      {}, vk::PrimitiveTopology::eTriangleList, false};

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

  {
    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format1d;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
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
  }

  framebuffer = device->createFramebufferUnique(vk::FramebufferCreateInfo{
      {}, *renderPass, 1, &(*imageView), extent.width, extent.height, 1});

  commandPoolUnique = device->createCommandPoolUnique(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       static_cast<uint32_t>(graphicsQueueFamilyIndex)});
  createVertexBuffer();
  createIndexBuffer();
  createStagingBuffer();

  auto bar = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
      *commandPoolUnique, vk::CommandBufferLevel::ePrimary, 1));
  renderBuffer = std::move(bar[0]);

  deviceQueue = device->getQueue(graphicsQueueFamilyIndex, 0);

  for (int i = 0; i < 4; ++i)
    clearValues.color.float32[i] = std::nan("");

  vk::ImageCreateInfo imgInfo;
  imgInfo.imageType = vk::ImageType::e2D;
  imgInfo.format = format1d;
  imgInfo.extent.width = width;
  imgInfo.extent.height = height;
  imgInfo.extent.depth = 1;
  imgInfo.arrayLayers = 1;
  imgInfo.mipLevels = 1;
  imgInfo.initialLayout = vk::ImageLayout::eUndefined;
  imgInfo.tiling = vk::ImageTiling::eLinear;
  imgInfo.usage = vk::ImageUsageFlagBits::eTransferDst;

  outputImage = device->createImageUnique(imgInfo);
  vk::MemoryRequirements reqs =
      device->getImageMemoryRequirements(*outputImage);
  vk::MemoryAllocateInfo outAllocInfo;
  outAllocInfo.allocationSize = reqs.size;
  outAllocInfo.memoryTypeIndex = findMemoryType(
      reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible |
                               vk::MemoryPropertyFlagBits::eHostCoherent);
  outputMem = device->allocateMemoryUnique(outAllocInfo);
  device->bindImageMemory(*outputImage, *outputMem, 0);

  vk::CommandBufferAllocateInfo allocInfo;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandPool = *commandPoolUnique;
  allocInfo.commandBufferCount = 1;
  auto foo = device->allocateCommandBuffersUnique(allocInfo);
  copyBackBuffer = std::move(foo[0]);

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
  vk::FenceCreateInfo info;
  fence = device->createFenceUnique(info);
}

void HeadlessInterpolator::run() {
  auto start = std::chrono::high_resolution_clock::now();
  int frames = 0;

  vk::CommandBuffer buffers[]{*copyBuffer, *renderBuffer, *copyBackBuffer};

  vk::SubmitInfo submitInfo;
  submitInfo.pSignalSemaphores = nullptr;
  submitInfo.pWaitSemaphores = nullptr;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pWaitDstStageMask = nullptr;
  submitInfo.commandBufferCount = 3;
  submitInfo.pCommandBuffers = buffers;

  while (true) {
    createPoints();
    auto extent = vk::Extent2D{width, height};
    auto beginInfo = vk::CommandBufferBeginInfo{};
    renderBuffer->begin(beginInfo);
    auto renderPassBeginInfo = vk::RenderPassBeginInfo{
        *renderPass, *framebuffer, vk::Rect2D{{0, 0}, extent}, 1, &clearValues};

    vk::DeviceSize offsets[] = {0, (vk::DeviceSize)PTS_SIZE};
    renderBuffer->bindVertexBuffers(0, 1, &*vertexBuffer.first, &offsets[0]);
    renderBuffer->bindIndexBuffer(*indexBuffer.first, 0,
                                  vk::IndexType::eUint32);

    renderBuffer->beginRenderPass(renderPassBeginInfo,
                                  vk::SubpassContents::eInline);
    renderBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

    renderBuffer->drawIndexed(nTris * 3, 1, 0, 0, 0);
    renderBuffer->endRenderPass();
    renderBuffer->end();

    device->resetFences(1, &*fence);
    deviceQueue.submit(submitInfo, fence.get());
    device->waitForFences(1, &*fence, true,
                          std::numeric_limits<uint64_t>::max());

    frames++;
    auto stop = std::chrono::high_resolution_clock::now();
    if (frames % 1000 == 0) {
      std::cout << "FPS: " << (frames * 1e9 / (stop - start).count())
                << std::endl;
    }

    vk::SubresourceLayout layout = device->getImageSubresourceLayout(
        *outputImage, {vk::ImageAspectFlagBits::eColor});
    const char *data = (const char *)device->mapMemory(
        *outputMem, layout.offset, layout.arrayPitch, vk::MemoryMapFlags{});
    (void)data;

#if 0   
    static int id = 0;
    std::ofstream res("foo" + std::to_string(id++ % 1000) + ".ppm",
                      std::ios_base::binary);
    for (uint32_t y = 0; y < height; ++y) {
      uint32_t *row = (uint32_t *)data;
      res.write((char *)row, sizeof(float) * width);
      data += layout.rowPitch;
    }
    res.close();
#endif
    device->unmapMemory(*outputMem);
  }
}

} // namespace vulkan_interpolator

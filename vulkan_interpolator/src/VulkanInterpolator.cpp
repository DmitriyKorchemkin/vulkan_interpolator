#include "vulkan_interpolator/VulkanInterpolator.hpp"

#include "shaders.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

#include <delaunator.hpp>

#define FENCE

static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                         int mods) {
  (void)scancode;
  (void)mods;
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

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
    throw "Failed to load vkCreateDebugUtilsMessengerEXT";  // check shouldn't
                                                            // be necessary
                                                            // (based on spec)
  CreateDebugUtilsMessengerEXTDispatchTable[instance] =
      reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(temp_fp);

  temp_fp = vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (!temp_fp)
    throw "Failed to load vkDestroyDebugUtilsMessengerEXT";  // check shouldn't
                                                             // be necessary
                                                             // (based on spec)
  DestroyDebugUtilsMessengerEXTDispatchTable[instance] =
      reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(temp_fp);

  temp_fp = vkGetInstanceProcAddr(instance, "vkSubmitDebugUtilsMessageEXT");
  if (!temp_fp)
    throw "Failed to load vkSubmitDebugUtilsMessageEXT";  // check shouldn't be
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

namespace vulkan_interpolator {

auto kernel2string(const std::string_view &view) {
  std::ostringstream kernel;
  for (auto &c : view)
    kernel << std::hex << std::setfill('0') << std::setw(2) << (uint32_t)c;
  return kernel.str();
}

auto register_shader(const std::string_view &code, vk::Device &device) {
  vk::ShaderModuleCreateInfo info;
  info.codeSize = code.size();
  info.pCode = reinterpret_cast<const uint32_t *>(code.data());
  return device.createShaderModuleUnique(info);
}
void Interpolator::createPoints() {
  std::uniform_real_distribution<float> u01(0, 1), u11(-1, 1);

  points.resize(N_PTS * 6);
  int idx = 0;
  std::vector<double> coords;
  for (auto &p : points) {

    p = (idx % 6) < 3 ? u11(rng)  : u01(rng);
    if (idx % 6 == 1) {
      coords.push_back(points[idx - 1]);
      coords.push_back(points[idx]);
    }
    ++idx;
  }
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
    for (int j = 2; j >=0; --j) {
    indicies[i*3+j]=i*3+j;// = d.triangles[i*3+j];
    }
//    std::cout << indicies[i] << ' ';
  }
    std::cout << std::endl;
  stageData();
}
uint32_t Interpolator::findMemoryType(
    uint32_t mask, const vk::MemoryPropertyFlags &properties) {
  auto props = physicalDevice.getMemoryProperties();
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((mask & (1 << i)) &&
        (props.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  }
  throw std::runtime_error("Incompatible memory type");
}
Interpolator::BufferMem Interpolator::createBuffer(
    size_t sz, const vk::BufferUsageFlags &usage,
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
void Interpolator::createStagingBuffer() {
  vk::DeviceSize size = PTS_SIZE + IDX_MAX_SIZE;
  stagingBuffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                               vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent);
//  device->mapMemory(*stagingBuffer.second, 0, size, vk::MemoryMapFlags{},
//                    &stagingData);


  vk::CommandBufferAllocateInfo allocInfo;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandPool = *commandPoolUnique;
  allocInfo.commandBufferCount = 1;

  auto foo  = device->allocateCommandBuffersUnique(allocInfo);
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
    copyBuffer->copyBuffer(*stagingBuffer.first, *indexBuffer.first, 1, regions+1);
    copyBuffer->end();
  

}

void Interpolator::stageData() {
  int IDX_SIZE = nTris * 3 * sizeof(int);
  device->mapMemory(*stagingBuffer.second, 0, PTS_SIZE+IDX_MAX_SIZE, vk::MemoryMapFlags{},
                    &stagingData);
  {
  memcpy(stagingData, points.data(), PTS_SIZE);
  }
  {
  memcpy((uint8_t*)stagingData + PTS_SIZE, indicies.data(), IDX_SIZE);
  }
  device->unmapMemory(*stagingBuffer.second);
}

void Interpolator::createVertexBuffer() {
  vertexBuffer = createBuffer(PTS_SIZE,
                              vk::BufferUsageFlagBits::eVertexBuffer |
                                  vk::BufferUsageFlagBits::eTransferDst,
                              vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void Interpolator::createIndexBuffer() {
  indexBuffer = createBuffer(IDX_MAX_SIZE,
                             vk::BufferUsageFlagBits::eIndexBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst,
                             vk::MemoryPropertyFlagBits::eDeviceLocal);
}

Interpolator::~Interpolator() {
//  device->unmapMemory(*stagingBuffer.second);
  glfwDestroyWindow(window);
  glfwTerminate();
}

Interpolator::Interpolator(const std::vector<size_t> &allowed_devices) {

  (void)allowed_devices;
  std::cout << "Vertex shader: (" << shaders::shader_vert_spv.size()
            << " bytes)" << kernel2string(shaders::shader_vert_spv)
            << std::endl;
  std::cout << "Fragment shader: (" << shaders::shader_frag_spv.size()
            << " bytes)" << kernel2string(shaders::shader_frag_spv)
            << std::endl;
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(width, height, "Vulkan 101", nullptr, nullptr);

  glfwSetKeyCallback(window, key_callback);

  vk::ApplicationInfo appInfo("Hello Triangle", VK_MAKE_VERSION(1, 0, 0),
                              "No Engine", VK_MAKE_VERSION(1, 0, 0),
                              VK_API_VERSION_1_0);

  auto glfwExtensionCount = 0u;
  auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> glfwExtensionsVector(
      glfwExtensions, glfwExtensions + glfwExtensionCount);
  glfwExtensionsVector.push_back("VK_EXT_debug_utils");
  auto layers =
      std::vector<const char *>{"VK_LAYER_LUNARG_standard_validation"};

  instance = vk::createInstanceUnique(
      vk::InstanceCreateInfo{{},
                             &appInfo,
                             static_cast<uint32_t>(layers.size()),
                             layers.data(),
                             static_cast<uint32_t>(glfwExtensionsVector.size()),
                             glfwExtensionsVector.data()});
  auto physicalDevices = instance->enumeratePhysicalDevices();

  physicalDevice = physicalDevices[0];
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

  VkSurfaceKHR surfaceTmp;
  glfwCreateWindowSurface(*instance, window, nullptr, &surfaceTmp);

  surface = vk::UniqueSurfaceKHR(surfaceTmp, *instance);

  auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

  size_t graphicsQueueFamilyIndex = std::distance(
      queueFamilyProperties.begin(),
      std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                   [](vk::QueueFamilyProperties const &qfp) {
                     return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
                   }));

  size_t presentQueueFamilyIndex = 0u;
  for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
    if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i),
                                            surface.get())) {
      presentQueueFamilyIndex = i;
    }
  }

  std::set<size_t> uniqueQueueFamilyIndices = {graphicsQueueFamilyIndex,
                                               presentQueueFamilyIndex};

  std::vector<uint32_t> FamilyIndices = {uniqueQueueFamilyIndices.begin(),
                                         uniqueQueueFamilyIndices.end()};

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

  float queuePriority = 0.0f;
  for (int queueFamilyIndex : uniqueQueueFamilyIndices) {
    queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{
        vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(queueFamilyIndex),
        1, &queuePriority});
  }

  const std::vector<const char *> deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  device = physicalDevice.createDeviceUnique(vk::DeviceCreateInfo(
      vk::DeviceCreateFlags(), queueCreateInfos.size(), queueCreateInfos.data(),
      0, nullptr, deviceExtensions.size(), deviceExtensions.data()));

  uint32_t imageCount = 2;

  struct SM {
    vk::SharingMode sharingMode;
    uint32_t familyIndicesCount;
    uint32_t *familyIndicesDataPtr;
  } sharingModeUtil{
      (graphicsQueueFamilyIndex != presentQueueFamilyIndex)
          ? SM{vk::SharingMode::eConcurrent, 2u, FamilyIndices.data()}
          : SM{vk::SharingMode::eExclusive, 0u,
               static_cast<uint32_t *>(nullptr)}};

  // needed for validation warnings
  auto formats = physicalDevice.getSurfaceFormatsKHR(*surface);

  auto format = vk::Format::eB8G8R8A8Unorm;
  auto extent = vk::Extent2D{width, height};

  vk::SwapchainCreateInfoKHR swapChainCreateInfo(
      {}, surface.get(), imageCount, format, vk::ColorSpaceKHR::eSrgbNonlinear,
      extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
      sharingModeUtil.sharingMode, sharingModeUtil.familyIndicesCount,
      sharingModeUtil.familyIndicesDataPtr,
      vk::SurfaceTransformFlagBitsKHR::eIdentity,
      vk::CompositeAlphaFlagBitsKHR::eOpaque, vk::PresentModeKHR::eImmediate,
      true, nullptr);

  swapChain = device->createSwapchainKHRUnique(swapChainCreateInfo);

  std::vector<vk::Image> swapChainImages =
      device->getSwapchainImagesKHR(swapChain.get());

  imageViews.reserve(swapChainImages.size());
  for (auto image : swapChainImages) {
    vk::ImageViewCreateInfo imageViewCreateInfo(
        vk::ImageViewCreateFlags(), image, vk::ImageViewType::e2D, format,
        vk::ComponentMapping{vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                             vk::ComponentSwizzle::eB,
                             vk::ComponentSwizzle::eA},
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    imageViews.push_back(device->createImageViewUnique(imageViewCreateInfo));
  }

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
  vertexBinding.stride = sizeof(float) * 6;
  vertexBinding.inputRate = vk::VertexInputRate::eVertex;

  vk::VertexInputAttributeDescription vertexAttributes[] = {
      {0, 0, vk::Format::eR32G32B32Sfloat, 0},
      {1, 0, vk::Format::eR32G32B32Sfloat, 3 * sizeof(float)}};

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
  vertexInputInfo.pVertexAttributeDescriptions = vertexAttributes;
  vertexInputInfo.pVertexBindingDescriptions = &vertexBinding;
  vertexInputInfo.vertexAttributeDescriptionCount = 2;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
#if 0
  auto vertexInputInfo =
      vk::PipelineVertexInputStateCreateInfo{{}, 0u, nullptr, 0u, nullptr};
#endif

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
                                format,
                                vk::SampleCountFlagBits::e1,
                                vk::AttachmentLoadOp::eClear,
                                vk::AttachmentStoreOp::eStore,
                                {},
                                {},
                                {},
                                vk::ImageLayout::ePresentSrcKHR};

  auto colourAttachmentRef =
      vk::AttachmentReference{0, vk::ImageLayout::eColorAttachmentOptimal};

  auto subpass = vk::SubpassDescription{{},
                                        vk::PipelineBindPoint::eGraphics,
                                        /*inAttachmentCount*/ 0,
                                        nullptr,
                                        1,
                                        &colourAttachmentRef};

  auto semaphoreCreateInfo = vk::SemaphoreCreateInfo{};
  imageAvailableSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);
  renderFinishedSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);

  auto subpassDependency =
      vk::SubpassDependency{VK_SUBPASS_EXTERNAL,
                            0,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            {},
                            vk::AccessFlagBits::eColorAttachmentRead |
                                vk::AccessFlagBits::eColorAttachmentWrite};

  renderPass = device->createRenderPassUnique(vk::RenderPassCreateInfo{
      {}, 1, &colorAttachment, 1, &subpass, 1, &subpassDependency});

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

  framebuffers = std::vector<vk::UniqueFramebuffer>(imageCount);
  for (size_t i = 0; i < imageViews.size(); i++) {
    framebuffers[i] = device->createFramebufferUnique(vk::FramebufferCreateInfo{
        {}, *renderPass, 1, &(*imageViews[i]), extent.width, extent.height, 1});
  }
  commandPoolUnique = device->createCommandPoolUnique(
      {{}, static_cast<uint32_t>(graphicsQueueFamilyIndex)});
  createVertexBuffer();
  createIndexBuffer();
  createStagingBuffer();


  commandBuffers =
      device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
          commandPoolUnique.get(), vk::CommandBufferLevel::ePrimary,
          framebuffers.size()));

  deviceQueue = device->getQueue(graphicsQueueFamilyIndex, 0);
  presentQueue = device->getQueue(presentQueueFamilyIndex, 0);

  for (size_t i = 0; i < commandBuffers.size(); i++) {
    auto beginInfo = vk::CommandBufferBeginInfo{};
    commandBuffers[i]->begin(beginInfo);
    vk::ClearValue clearValues{};
    auto renderPassBeginInfo =
        vk::RenderPassBeginInfo{renderPass.get(), framebuffers[i].get(),
                                vk::Rect2D{{0, 0}, extent}, 1, &clearValues};

    vk::DeviceSize offsets[] = {0, (vk::DeviceSize)PTS_SIZE};
    commandBuffers[i]->bindVertexBuffers(0, 1, &*vertexBuffer.first, &offsets[0]);
    commandBuffers[i]->bindIndexBuffer(*indexBuffer.first, 0, vk::IndexType::eUint32);
        
    commandBuffers[i]->beginRenderPass(renderPassBeginInfo,
                                       vk::SubpassContents::eInline);
    commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                                    pipeline.get());


    commandBuffers[i]->drawIndexed(nTris, 1, 0, 0, 0);
    commandBuffers[i]->endRenderPass();
    commandBuffers[i]->end();
  }
}

void Interpolator::run() {
  vk::UniqueFence fence;
  vk::FenceCreateInfo info;
  fence = device->createFenceUnique(info);
  auto start = std::chrono::high_resolution_clock::now();
  int frames = 0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    auto imageIndex = device->acquireNextImageKHR(
        swapChain.get(), std::numeric_limits<uint64_t>::max(),
        imageAvailableSemaphore.get(), {});

    vk::PipelineStageFlags waitStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;


//    std::cout << "Copying" << std::endl;
#if 0
    vk::SubmitInfo copyInfo;
    copyInfo.commandBufferCount = 1;
    copyInfo.pCommandBuffers = &*copyBuffer;
    deviceQueue.submit(copyInfo, {});
    deviceQueue.waitIdle();
#endif
#if 0
    std::cout << "Rendering"<<std::endl;
    std::cout << "Pts:" << std::endl;
    for (int i = 0; i < N_TRI; ++i) {
      std::cout << "---------------------------\n";
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k)
        std::cout << points[i*6*3 + j*6+k] << ' ';
        std::cout << "| ";
        for (int k = 3; k < 6; ++k)
        std::cout << points[i*6*3 + j*6+k] << ' ';
        std::cout << '\n';
      }
    }
      std::cout << "---------------------------\n";
      std::cout << "---------------------------\n";
      for (auto& i: indicies)
        std::cout << i << ' ';
      std::cout << std::endl;
#endif


#if 0
    auto submitInfo = vk::SubmitInfo{1,
                                     &imageAvailableSemaphore.get(),
                                     &waitStageMask,
                                     1,
                                     &commandBuffers[imageIndex.value].get(),
                                     1,
                                     &renderFinishedSemaphore.get()};
#else
    vk::CommandBuffer buffers[]{ *copyBuffer, *commandBuffers[imageIndex.value]};
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 2;
    submitInfo.pCommandBuffers = buffers;
    submitInfo.pSignalSemaphores = &*renderFinishedSemaphore;
    submitInfo.pWaitSemaphores = &*imageAvailableSemaphore;
    submitInfo.waitSemaphoreCount=1;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pWaitDstStageMask = &waitStageMask;
#endif

   
#ifdef FENCE
    device->resetFences(1, &*fence);
    deviceQueue.submit(submitInfo, *fence);
#else
    deviceQueue.submit(submitInfo, {});
#endif

    auto presentInfo = vk::PresentInfoKHR{1, &renderFinishedSemaphore.get(), 1,
                                          &swapChain.get(), &imageIndex.value};
    presentQueue.presentKHR(presentInfo);

    createPoints();
#ifdef FENCE
    device->waitForFences(1, &*fence, true,
                          std::numeric_limits<uint64_t>::max());
#else
    device->waitIdle();
#endif
    ++frames;
//    break;
    auto stop = std::chrono::high_resolution_clock::now();
    if (frames % 10 == 0) {
      std::cout << "FPS: " << (frames * 1e9 / (stop - start).count())
                << std::endl;
    }
//    break;
  }
}


}  // namespace vulkan_interpolator

#include "vulkan_interpolator/VulkanInterpolator.hpp"

#include "shaders.hpp"

#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <chrono>

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

  auto physicalDevice = physicalDevices[0];
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
      vk::CompositeAlphaFlagBitsKHR::eOpaque, vk::PresentModeKHR::eFifoRelaxed, true,
      nullptr);

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

  auto vertexInputInfo =
      vk::PipelineVertexInputStateCreateInfo{{}, 0u, nullptr, 0u, nullptr};

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
      /*frontFace*/ vk::FrontFace::eCounterClockwise,
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

    commandBuffers[i]->beginRenderPass(renderPassBeginInfo,
                                       vk::SubpassContents::eInline);
    commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                                    pipeline.get());
    commandBuffers[i]->draw(3, 1, 0, 0);
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

    auto submitInfo = vk::SubmitInfo{1,
                                     &imageAvailableSemaphore.get(),
                                     &waitStageMask,
                                     1,
                                     &commandBuffers[imageIndex.value].get(),
                                     1,
                                     &renderFinishedSemaphore.get()};

//#define FENCE
#ifdef FENCE
    device->resetFences(1, &*fence);
    deviceQueue.submit(submitInfo, *fence);
#else
    deviceQueue.submit(submitInfo, {});
#endif

    auto presentInfo = vk::PresentInfoKHR{1, &renderFinishedSemaphore.get(), 1,
                                          &swapChain.get(), &imageIndex.value};
    presentQueue.presentKHR(presentInfo);

#ifdef FENCE
    device->waitForFences(1, &*fence, true, std::numeric_limits<uint64_t>::max());
#else
    device->waitIdle();
#endif
    ++frames;
   auto stop = std::chrono::high_resolution_clock::now();
   if (frames%1000==0) {
     std::cout << "FPS: " <<  (frames * 1e9 / (stop - start).count()) << std::endl;
   }
  }
}

Interpolator::~Interpolator() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

}  // namespace vulkan_interpolator

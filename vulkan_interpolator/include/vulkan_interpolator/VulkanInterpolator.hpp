#ifndef VULKAN_INTERPOLATOR
#define VULKAN_INTERPOLATOR

#include <vector>
#include <vulkan/vulkan.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace vulkan_interpolator {

struct Interpolator {
    static const uint32_t width = 800;
    static const uint32_t height = 600;
    GLFWwindow* window;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueDevice device;
    vk::UniqueSwapchainKHR swapChain;
    std::vector<vk::UniqueImageView> imageViews;
    vk::UniqueShaderModule vertexShaderModule, fragmentShaderModule;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueSemaphore imageAvailableSemaphore, renderFinishedSemaphore;
    vk::UniqueRenderPass renderPass;
    vk::UniquePipeline pipeline;
    std::vector<vk::UniqueFramebuffer> framebuffers;
    vk::UniqueCommandPool commandPoolUnique;
    vk::Queue presentQueue, deviceQueue;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;

  Interpolator(const std::vector<size_t> &devices);
  ~Interpolator();
  void run();

private:
  std::vector<const char *> required_extensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};
};

} // namespace vulkan_interpolator

#endif

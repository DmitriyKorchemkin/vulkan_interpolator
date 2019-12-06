#include <iostream>
#include <vulkan/vulkan.hpp>

struct Foobar {
  vk::Instance instance;
  std::vector<vk::PhysicalDevice> devices;

  Foobar();
  ~Foobar();
};

Foobar::Foobar() {
  vk::ApplicationInfo info;
  info.pApplicationName = "InterpolatorTest";
  info.pEngineName = "vulkan_interpolator";
  info.apiVersion = VK_API_VERSION_1_0;

  vk::InstanceCreateInfo instanceCreateInfo;
  instanceCreateInfo.pApplicationInfo = &info;
  instance = vk::createInstance(instanceCreateInfo);
  devices = instance.enumeratePhysicalDevices();
  std::cout << "Found " << devices.size() << " vulkan-devices" << std::endl;
}

Foobar::~Foobar() { instance.destroy(); }

int main(int argc, char **argv) {
  Foobar foobar;

  return 0;
}

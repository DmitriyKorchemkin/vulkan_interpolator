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

  for (auto &dev : devices) {
    auto props = dev.getProperties();
    auto mem = dev.getMemoryProperties();
    std::cout << props.deviceID << " " << props.deviceName << " "
              << props.limits.maxMemoryAllocationCount / 1024. / 1024 / 1024
              << "GB allocation" << std::endl;
    for (int i = 0; i < mem.memoryHeapCount; ++i)
      std::cout << "Heap: " << mem.memoryHeaps[i].size/1024/1024/1024 << std::endl;
  }
}

Foobar::~Foobar() { instance.destroy(); }

int main(int argc, char **argv) {
  Foobar foobar;

  return 0;
}

#include <vulkan_interpolator/VulkanInterpolatorHeadless.hpp>

#include <iostream>


int main(int argc, char **argv) {
  if (argc > 1) {
    std::cout << "Unrecognized options: ";
    for (int i = 1; i < argc; ++i) std::cout << argv[i] << " ";
    std::cout << std::endl;
    return -1;
  }

  vulkan_interpolator::HeadlessInterpolator interpolator({});
  interpolator.run();

  return 0;
}

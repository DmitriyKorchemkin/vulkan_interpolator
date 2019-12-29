#include <vulkan_interpolator/VulkanInterpolatorHeadless.hpp>

#include <iostream>


int main(int argc, char **argv) {
  if (argc > 2) {
    std::cout << "Unrecognized options: ";
    for (int i = 1; i < argc; ++i) std::cout << argv[i] << " ";
    std::cout << std::endl;
    return -1;
  }

  size_t device = 0;
  if (argc == 2)
    device = std::atoi(argv[1])%4;

  vulkan_interpolator::HeadlessInterpolator interpolator({device});
  interpolator.run();

  return 0;
}

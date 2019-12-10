#include <vulkan_interpolator/VulkanInterpolator.hpp>

#include <iostream>


int main(int argc, char **argv) {
  if (argc > 1) {
    std::cout << "Unrecognized options: ";
    for (int i = 1; i < argc; ++i) std::cout << argv[i] << " ";
    std::cout << std::endl;
    return -1;
  }

  vulkan_interpolator::Interpolator interpolator({});
  interpolator.run();

  return 0;
}

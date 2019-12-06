#ifndef VULKAN_INTERPOLATOR
#define VULKAN_INTERPOLATOR

#include <vector>

namespace vulkan_interpolator {

struct Interpolator {
  Interpolator(const std::vector<int> &devices);

private:
  std::vector<int> allocatedDevices;
};

} // namespace vulkan_interpolator

#endif

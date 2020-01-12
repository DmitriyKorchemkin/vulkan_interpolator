#include "vulkan_interpolator/VulkanInterpolator.hpp"
#include "vulkan_interpolator/VulkanInterpolatorHeadless.hpp"
#include <vulkan/vulkan.hpp>

namespace vulkan_interpolator {
vk::Instance &GetInstance() {
  static vk::ApplicationInfo appInfo(
      "vulkan_interpolator", VK_MAKE_VERSION(1, 0, 0), "No Engine",
      VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0);
  static auto instance = vk::createInstanceUnique(
      vk::InstanceCreateInfo{{}, &appInfo, 0, nullptr, 0, nullptr});

  return *instance;
}

Interpolator::Interpolator(const size_t interpolatorsPerDevice,
                           const std::vector<size_t> &devices_,
                           const InterpolationOptions &options) {
  auto devices = devices_;
  if (!devices.size()) {
    auto nDevices = GetDeviceNumber();
    for (size_t i = 0; i < nDevices; ++i)
      devices.push_back(i);
  }
  size_t totalInterpolators = devices.size() * interpolatorsPerDevice;
  auto &instance = GetInstance();
  for (size_t i = 0; i < totalInterpolators; ++i) {
    auto interpolator = new HeadlessInterpolator({i / interpolatorsPerDevice},
                                                 options, &instance);
    interpolators.emplace_back(interpolator);
    freeInterpolators.push_back(interpolator);
  }
}

void Interpolator::interpolate(const int nPoints, const float *points,
                               const float *values, const int width,
                               const int height, const int stride_bytes,
                               float *output, float dt, float db, float dl,
                               float dr, float sx, float sy) {
  std::vector<int> indicies;
  PrepareInterpolation(nPoints, points, indicies);

  interpolate(nPoints, points, values, indicies.size() / 3, indicies.data(),
              width, height, stride_bytes, output, dt, db, dl, dr, sx, sy);
}

// Just rasterize
void Interpolator::interpolate(const int nPoints, const float *points,
                               const float *values, const int nTriangles,
                               const int *indicies, const int width,
                               const int height, const int stride_bytes,
                               float *output, float dt, float db, float dl,
                               float dr, float sx, float sy) {
  std::unique_lock<std::mutex> lock(mutex);

  HeadlessInterpolator *interpolator;
  if (!freeInterpolators.size()) {
    // wait till interpolator is available
    interpolatorAvailable.wait(lock,
                               [&]() { return freeInterpolators.size(); });
  }
  interpolator = freeInterpolators.back();
  freeInterpolators.pop_back();
  lock.unlock();

  interpolator->interpolate(nPoints, points, values, nTriangles, indicies,
                            width, height, stride_bytes, output, dt, db, dl, dr,
                            sx, sy);

  lock.lock();
  freeInterpolators.push_back(interpolator);
  interpolatorAvailable.notify_one();
}

// Compute delaunay triangulation
void Interpolator::PrepareInterpolation(const int nPoints, const float *points,
                                        std::vector<int> &indicies) {
  HeadlessInterpolator::PrepareInterpolation(nPoints, points, indicies);
}

// Gets number of vulkan devices
size_t Interpolator::GetDeviceNumber() {
  return GetInstance().enumeratePhysicalDevices().size();
}

} // namespace vulkan_interpolator

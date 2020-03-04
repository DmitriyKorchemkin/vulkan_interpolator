/******************************************************************************
 * Vulkan-Interpolator: interpolator/rasterizer based on Vulkan API
 * 
 * Copyright (c) 2019-2020 Dmitriy Korchemkin
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/
#include "vulkan_interpolator/Interpolator.hpp"
#include "vulkan_interpolator/HeadlessInterpolator.hpp"
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
  // auto &instance = GetInstance();
  for (size_t i = 0; i < totalInterpolators; ++i) {
    auto interpolator = new HeadlessInterpolator(
        {i / interpolatorsPerDevice}, options, nullptr); // &instance);
    interpolators.emplace_back(interpolator);
    freeInterpolators.push_back(interpolator);
  }
}

Interpolator::~Interpolator() {
  for (auto &i : interpolators)
    delete i;
}

void Interpolator::interpolate(const int nPoints, const float *points,
                               const float *values, const int width,
                               const int height, const int stride_bytes,
                               float *output) {
  std::vector<int> indicies;
  PrepareInterpolation(nPoints, points, 2, indicies);

  interpolate(nPoints, points, values, indicies.size() / 3, indicies.data(),
              width, height, stride_bytes, output);
}

void Interpolator::interpolate(const int nPoints,
                               const float *points_and_values, const int width,
                               const int height, const int stride_bytes,
                               float *output) {
  std::vector<int> indicies;
  PrepareInterpolation(nPoints, points_and_values, 3, indicies);

  interpolate(nPoints, points_and_values, indicies.size() / 3, indicies.data(),
              width, height, stride_bytes, output);
}

// Just rasterize
void Interpolator::interpolate(const int nPoints, const float *points,
                               const float *values, const int nTriangles,
                               const int *indicies, const int width,
                               const int height, const int stride_bytes,
                               float *output) {
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
                            width, height, stride_bytes, output);

  lock.lock();
  freeInterpolators.push_back(interpolator);
  interpolatorAvailable.notify_one();
}

// Just rasterize
void Interpolator::interpolate(const int nPoints,
                               const float *points_and_values,
                               const int nTriangles, const int *indicies,
                               const int width, const int height,
                               const int stride_bytes, float *output) {
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

  interpolator->interpolate(nPoints, points_and_values, nTriangles, indicies,
                            width, height, stride_bytes, output);

  lock.lock();
  freeInterpolators.push_back(interpolator);
  interpolatorAvailable.notify_one();
}

// Compute delaunay triangulation
void Interpolator::PrepareInterpolation(const int nPoints, const float *points,
                                        int stride,
                                        std::vector<int> &indicies) {
  HeadlessInterpolator::PrepareInterpolation(nPoints, points, stride, indicies);
}

// Gets number of vulkan devices
size_t Interpolator::GetDeviceNumber() {
  return GetInstance().enumeratePhysicalDevices().size();
}

} // namespace vulkan_interpolator

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
#ifndef VULKAN_INTERPOLATOR_INTERPOLATOR_HPP
#define VULKAN_INTERPOLATOR_INTERPOLATOR_HPP

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace vulkan_interpolator {

struct HeadlessInterpolator;
struct InterpolationOptions {
  int heightPreallocated = 2000;
  int widthPreallocated = 2000;
  int pointsPreallocated = 20000;
  int indiciesPreallocated = 30000;
};

struct Interpolator {
  // Note: if devices is empty -- all available devices are being used
  Interpolator(const size_t interpolatorsPerDevice = 8,
               const std::vector<size_t> &devices = {},
               const InterpolationOptions &options = InterpolationOptions());
  // Mimicks HeadlessInterpolator (and, effectively, hides all vulkan-related
  // stuff) Triangulate and rasterize
  void interpolate(const int nPoints, const float *points, const float *values,
                   const int width, const int height, const int stride_bytes,
                   float *output);
  void interpolate(const int nPoints, const float *points_and_values,
                   const int width, const int height, const int stride_bytes,
                   float *output);
  // Just rasterize
  void interpolate(const int nPoints, const float *points, const float *values,
                   const int nTriangles, const int *indicies, const int width,
                   const int height, const int stride_bytes, float *output);
  void interpolate(const int nPoints, const float *points_and_values,
                   const int nTriangles, const int *indicies, const int width,
                   const int height, const int stride_bytes, float *output);

  // Compute delaunay triangulation
  static void PrepareInterpolation(const int nPoints, const float *points,
                                   int stride, std::vector<int> &indicies);
  // Gets number of vulkan devices
  static size_t GetDeviceNumber();

  ~Interpolator();

private:
  Interpolator(const Interpolator &) = delete;
  Interpolator &operator=(const Interpolator &) = delete;
  std::mutex mutex;
  std::condition_variable interpolatorAvailable;
  std::vector<HeadlessInterpolator *> freeInterpolators;
  std::vector<HeadlessInterpolator *> interpolators;
};

} // namespace vulkan_interpolator

#endif

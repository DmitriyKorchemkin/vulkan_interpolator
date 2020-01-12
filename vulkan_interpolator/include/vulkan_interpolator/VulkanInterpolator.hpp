#ifndef VULKAN_INTERPOLATOR
#define VULKAN_INTERPOLATOR

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace vulkan_interpolator {

struct HeadlessInterpolator;
struct InterpolationOptions {
  int heightPreallocated = 1000;
  int widthPreallocated = 1000;
  int pointsPreallocated = 10000;
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
                   float *output, float dt, float db, float dl, float dr,
                   float sx, float sy);
  // Just rasterize
  void interpolate(const int nPoints, const float *points, const float *values,
                   const int nTriangles, const int *indicies, const int width,
                   const int height, const int stride_bytes, float *output,
                   float dt, float db, float dl, float dr, float sx, float sy);

  // Compute delaunay triangulation
  static void PrepareInterpolation(const int nPoints, const float *points,
                                   std::vector<int> &indicies);
  // Gets number of vulkan devices
  static size_t GetDeviceNumber();

private:
  std::mutex mutex;
  std::condition_variable interpolatorAvailable;
  std::vector<HeadlessInterpolator *> freeInterpolators;
  std::vector<std::unique_ptr<HeadlessInterpolator>> interpolators;
};

} // namespace vulkan_interpolator

#endif

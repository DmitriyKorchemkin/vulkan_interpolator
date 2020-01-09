#include <vulkan_interpolator/VulkanInterpolatorHeadless.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc > 2) {
    std::cout << "Unrecognized options: ";
    for (int i = 1; i < argc; ++i) std::cout << argv[i] << " ";
    std::cout << std::endl;
    return -1;
  }

  size_t device = 0;
  if (argc == 2) device = std::atoi(argv[1]) % 4;
  vulkan_interpolator::HeadlessInterpolator interpolator({device});

  std::mt19937 rng;
  std::uniform_int_distribution<int> rwh(1000, 5000), rn(10, 10000);
  std::vector<float> data;
  std::uniform_real_distribution<float> runif(-1.f, 1.f);

  for (int h = 0; h < 30; ++h) {
    const int N = rn(rng);

    int width = rwh(rng);
    int height = rwh(rng);
    std::uniform_real_distribution<float> rw(0, width);
    std::uniform_real_distribution<float> rh(0, height);

    std::vector<float> points(2 * N), values(N);
    for (int i = 0; i < N; ++i) {
      points[i * 2] = rw(rng);
      points[i * 2 + 1] = rh(rng);
      values[i] = runif(rng);
    }
    data.resize(width * height);

    interpolator.interpolate(N, points.data(), values.data(), width, height,
                             width * sizeof(float), data.data());
    std::cout << "==================" << std::endl;
  }

  std::ofstream out("out.bin", std::ios_base::binary);
  out.write(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));

  return 0;
}

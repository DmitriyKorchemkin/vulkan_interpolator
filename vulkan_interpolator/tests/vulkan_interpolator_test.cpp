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
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>

#include <vulkan_interpolator/Interpolator.hpp>

TEST(VulkanInterpolator, InterpolationCorrectness) {
  std::mt19937 rng;
  std::uniform_int_distribution<int> rwh(1024, 8192), rn(10, 100000);
  std::uniform_real_distribution<float> runif(-1.f, 1.f);

  vulkan_interpolator::Interpolator interpolator;
  for (int i = 0; i < 100; ++i) {
    const int N = rn(rng);

    int width = rwh(rng);
    int height = rwh(rng);
    std::uniform_int_distribution<int> rw(0, width - 1);
    std::uniform_int_distribution<int> rh(0, height - 1);

    std::vector<float> points(2 * N), values(N);
    for (int i = 0; i < N; ++i) {
      points[i * 2] = rw(rng);
      points[i * 2 + 1] = rh(rng);
      values[i] = runif(rng);
    }

    std::vector<int> tris;
    vulkan_interpolator::Interpolator::PrepareInterpolation(N, points.data(),
                                                            tris);
    std::vector<float> data(width * height),
        data_golden(width * height, std::nan(""));
    auto ts_a = std::chrono::high_resolution_clock::now();
    interpolator.interpolate(N, points.data(), values.data(), tris.size() / 3,
                             tris.data(), width, height, width * sizeof(float),
                             data.data());
    auto ts_b = std::chrono::high_resolution_clock::now();
    int total = 0, totalIn = 0;
    double misfit = 0.;
    for (size_t tri = 0; tri < tris.size(); tri += 3) {
      Eigen::AlignedBox2d box2f;
      Eigen::Matrix3d M;
      Eigen::Vector3d b;

      for (int iii = 0; iii < 3; ++iii) {
        M.col(iii) = Eigen::Vector2d(points[tris[tri + iii] * 2],
                                     points[tris[tri + iii] * 2 + 1])
                         .homogeneous();
        b[iii] = values[tris[tri + iii]];
        box2f.extend(M.col(iii).template head<2>());
      }

      Eigen::Matrix3d Mi = M.inverse();

      for (int y = std::max(0., std::floor(box2f.min().y()));
           y <= std::min(height - 1.0, std::ceil(box2f.max().y())); ++y) {
        for (int x = std::max(0., std::floor(box2f.min().x()));
             x <= std::min(width - 1.0, std::ceil(box2f.max().x())); ++x) {
          Eigen::Vector3d pt = Eigen::Vector2d(x, y).homogeneous();
          Eigen::Vector3d gamma = Mi * pt;
          if (gamma.array().maxCoeff() >= 1 || gamma.array().minCoeff() <= 0.) {
            continue;
          }
          totalIn++;
          data_golden[y * width + x] = gamma.dot(b);
          if (std::isnan(data[y * width + x]))
            continue;
          double diff = gamma.dot(b) - data[y * width + x];
          misfit += diff * diff;
          total++;
        }
      }
    }
    auto ts_c = std::chrono::high_resolution_clock::now();
    std::cout << "GPU is "
              << double((ts_c - ts_b).count()) / (ts_b - ts_a).count()
              << "x faster" << std::endl;
    misfit = std::sqrt(misfit / total);
    static double minRatio = 1., maxMisfit = 0.;
    minRatio = std::min(minRatio, double(total) / totalIn);
    maxMisfit = std::max(maxMisfit, misfit);
    EXPECT_GT(total, 0.999 * totalIn);
    EXPECT_LT(misfit, 1e-6);
  }
}

TEST(VulkanInterpolator, ParallelBurnTest) {
  vulkan_interpolator::Interpolator interpolator;

  std::mt19937 rng_global;
  std::vector<std::thread> workers;
  std::atomic<int> alive(0);
  int N = 100000;
  auto start_all = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    while (alive > 200)
      std::this_thread::yield();

    const auto seed = rng_global();
    workers.emplace_back([&interpolator, &alive,seed]() {
      ++alive;
      std::mt19937 rng(seed);
      std::uniform_int_distribution<int> rwh(2048, 4096), rn(1000, 10000);
      std::uniform_real_distribution<float> runif(-1.f, 1.f);

      const int N = rn(rng);

      int width = rwh(rng);
      int height = rwh(rng);
      std::uniform_int_distribution<int> rw(0, width - 1);
      std::uniform_int_distribution<int> rh(0, height - 1);

      std::vector<float> points(2 * N), values(N);
      for (int i = 0; i < N; ++i) {
        points[i * 2] = rw(rng);
        points[i * 2 + 1] = rh(rng);
        values[i] = runif(rng);
      }
      std::vector<float> data(width * height);
      auto start = std::chrono::high_resolution_clock::now();
      interpolator.interpolate(N, points.data(), values.data(), width, height,
                               width * sizeof(float), data.data());
      auto stop = std::chrono::high_resolution_clock::now();
      std::cout << (1e9 / (stop - start).count()) << "FPS" << std::endl;
      --alive;
    });
  }
  for (auto &worker : workers)
    worker.join();
  auto stop_all = std::chrono::high_resolution_clock::now();
  std::cout << (1e9 / (stop_all - start_all).count()) << "FPS-all" << std::endl;
}

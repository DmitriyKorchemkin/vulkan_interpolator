#include <vulkan_interpolator/VulkanInterpolatorHeadless.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include <tbb/tbb.h>

int main(int argc, char **argv) {
  if (argc > 2) {
    std::cout << "Unrecognized options: ";
    for (int i = 1; i < argc; ++i)
      std::cout << argv[i] << " ";
    std::cout << std::endl;
    return -1;
  }

  size_t device = 0;
  if (argc == 2)
    device = std::atoi(argv[1]) % 4;
  vulkan_interpolator::HeadlessInterpolator interpolator({device});

  std::mt19937 rng;
  std::uniform_int_distribution<int> rwh(1000, 4000), rn(3000, 4000);
  std::uniform_real_distribution<float> runif(-1.f, 1.f);

  for (int h = 0; h < 4999; ++h) {
    const int N = rn(rng);

    int width = rwh(rng);
    int height = rwh(rng);
    std::uniform_real_distribution<float> rw(0, width);
    std::uniform_real_distribution<float> rh(0, height);

    std::vector<float> points(2 * N), values(N);
    for (int i = 0; i < N; ++i) {
      points[i * 2] = rw(rng);
      points[i * 2 + 1] = rh(rng);
      std::cout << points[i * 2] << " " << points[i * 2 + 1] << '\n';
      values[i] = runif(rng);
    }

    std::vector<int> tris;
    vulkan_interpolator::HeadlessInterpolator::PrepareInterpolation(
        N, points.data(), tris);
    std::cout << "Tris size: " << tris.size() << '\n';

    std::vector<float> delta_values;
    float signs[] = {1.f};
    float deltas[] = {
        -1.f, -0.5f, 0.f, 0.5f,
        1.f}; // 0.f};  // {-1.5f, -1.f, -.5f, 0.f, .5f, 1.f, 1.5f};
    float minDeltas[6];
    float minDelta = 1e100;
    int cntD = 0;
    for (auto &sx : signs)
      for (auto &sy : signs)
        for (auto &dt : deltas)
          for (auto &db : deltas)
            for (auto &dl : deltas)
              for (auto &dr : deltas) {
                delta_values.push_back(dt);
                delta_values.push_back(db);
                delta_values.push_back(dl);
                delta_values.push_back(dr);
                delta_values.push_back(sx);
                delta_values.push_back(sy);
                ++cntD;
              }
    std::vector<double> misfits(cntD);
    std::vector<int> totals(cntD), totalsIn(cntD);
#if 0
    tbb::parallel_for(tbb::blocked_range<int>(0, cntD), [&](const auto &range) {
      std::vector<float> data(width * height),
          data_golden(width * height, std::nan(""));
      for (int id = range.begin(); id != range.end(); ++id) {
#else
    std::vector<float> data(width * height),
        data_golden(width * height, std::nan(""));
    for (int id = 0; id < cntD; ++id) {
#endif
    data_golden.clear();
    data_golden.resize(data.size(), std::nan(""));
    const float dt = delta_values[id * 6], db = delta_values[id * 6 + 1],
                dl = delta_values[id * 6 + 2], dr = delta_values[id * 6 + 3],
                sx = delta_values[id * 6 + 4], sy = delta_values[id * 6 + 5];
    auto ts_a = std::chrono::high_resolution_clock::now();
    interpolator.interpolate(N, points.data(), values.data(), tris.size() / 3,
                             tris.data(), width, height, width * sizeof(float),
                             data.data(), dt, db, dl, dr, sx, sy);
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

#if 0
        static std::mutex mutex;
        std::lock_guard lock(mutex);
        std::ofstream of("out.m", std::ios_base::app);
        of << "M" << id << " = [";
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            of << data[y * width + x]
               << (x + 1 == width ? y + 1 == height ? ']' : ';' : ',');
          }
        }
        of << ";\n";
        of << "G" << id << " = [";
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            of << data_golden[y * width + x]
               << (x + 1 == width ? y + 1 == height ? ']' : ';' : ',');
          }
        }
        of << ";\n";
#endif
        std::cout << '#' << std::flush;
        misfit = std::sqrt(misfit / total);
        misfits[id] = misfit;
        totals[id] = total;
        totalsIn[id] = totalIn;
      }
#if 0
    });
#endif
      for (int id = 0; id < cntD; ++id) {
        const float dt = delta_values[id * 6], db = delta_values[id * 6 + 1],
                    dl = delta_values[id * 6 + 2],
                    dr = delta_values[id * 6 + 3],
                    sx = delta_values[id * 6 + 4],
                    sy = delta_values[id * 6 + 5];
        const double misfit = misfits[id];
        if (minDelta > misfit) {
          minDelta = misfit;
          minDeltas[0] = sx;
          minDeltas[1] = sy;
          minDeltas[2] = dt;
          minDeltas[3] = db;
          minDeltas[4] = dl;
          minDeltas[5] = dr;
        }
        std::cout << sx << " " << sy << " " << dt << " " << db << " " << dl
                  << " " << dr << " " << misfit << " @ " << totals[id] << "/"
                  << totalsIn[id] << "px" << std::endl;
      }

      std::cout << "Best deltas: " << minDeltas[0] << " " << minDeltas[1] << " "
                << minDeltas[2] << " " << minDeltas[3] << " " << minDeltas[4]
                << " " << minDeltas[5] << " @" << minDelta << std::endl;
      std::cout << "==================" << std::endl;
  }
#if 0
  std::ofstream out("out.bin", std::ios_base::binary);
  out.write(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
#endif
  return 0;
}

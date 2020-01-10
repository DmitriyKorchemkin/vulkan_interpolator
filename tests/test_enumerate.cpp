#include <vulkan_interpolator/VulkanInterpolatorHeadless.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include <tbb/tbb.h>

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
  std::uniform_int_distribution<int> rwh(1000, 2000), rn(10, 15);
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

    std::vector<int> tris;
    vulkan_interpolator::HeadlessInterpolator::PrepareInterpolation(
        N, points.data(), tris);

    std::vector<float> delta_values;
    float signs[] = {-1.f, 1.f};
    float deltas[] = {-1.5f, -1.f, -.5f, 0.f, .5f, 1.f, 1.5f};
    float minDeltas[6];
    float minDelta = 1e100;
    int cntD = 0;
    for (auto& sx: signs)
      for (auto& sy: signs)
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

    tbb::parallel_for(tbb::blocked_range<int>(0, cntD), [&](const auto &range) {
  std::vector<float> data(width*height);
      for (int id = range.begin(); id != range.end(); ++id) {
        const float dt = delta_values[id * 6], db = delta_values[id * 6 + 1],
                    dl = delta_values[id * 6 + 2],
                    dr = delta_values[id * 6 + 3], sx = delta_values[id * 6 + 4], sy = delta_values[id * 6 + 5];
        interpolator.interpolate(N, points.data(), values.data(), width, height,
                                 width * sizeof(float), data.data(), dt, db, dl,
                                 dr, sx, sy);
        int total = 0, totalIn = 0;
        double misfit = 0.;
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            Eigen::Vector3d pt = Eigen::Vector2d(x, y).homogeneous();
            for (size_t tri = 0; tri < tris.size(); tri += 3) {
              Eigen::Matrix3d M;
              Eigen::Vector3d b;
              for (int iii = 0; iii < 3; ++iii) {
                M.row(iii) = Eigen::Vector2d(points[tris[tri + iii] * 2],
                                             points[tris[tri + iii] * 2 + 1])
                                 .homogeneous()
                                 .transpose();
                b[iii] = values[tris[tri + iii]];
              }
              Eigen::Vector3d c = M.lu().solve(b);
              Eigen::Vector3d gamma = M.transpose().lu().solve(pt);
              if (gamma.array().maxCoeff() >= 1 ||
                  gamma.array().minCoeff() <= 0.)
                continue;
              totalIn++;
              if (std::isnan(data[y * width + x]))
                continue;
              double diff = pt.dot(c) - data[y * width + x];
              misfit += diff * diff;
              total++;
            }
          }
        }
        std::cout << '#' << std::flush;
        misfit = std::sqrt(misfit / total);
        misfits[id] = misfit;
        totals[id] = total;
        totalsIn[id] = totalIn;
      }
    });
    for (int id = 0; id < cntD; ++id) {
      const float dt = delta_values[id * 6], db = delta_values[id * 6 + 1],
                  dl = delta_values[id * 6 + 2], dr = delta_values[id * 6 + 3], sx= delta_values[id*6+4], sy = delta_values[id * 6 + 5];
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
      std::cout << sx << " " << sy << " " << dt << " " << db << " " << dl << " " << dr << " " << misfit
                << " @ " << totals[id] << "/" << totalsIn[id] << "px"
                << std::endl;
    }

    std::cout << "Best deltas: " << minDeltas[0] << " " << minDeltas[1] << " "
              << minDeltas[2] << " " << minDeltas[3] << " " << minDeltas[4] << " " << minDeltas[5] << " @" << minDelta
              << std::endl;
    std::cout << "==================" << std::endl;
  }
#if 0
  std::ofstream out("out.bin", std::ios_base::binary);
  out.write(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
#endif
  return 0;
}

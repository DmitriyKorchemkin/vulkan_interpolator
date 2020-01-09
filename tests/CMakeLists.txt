
find_package(Eigen3 REQUIRED CONFIG)

add_executable(test_devices test_enumerate.cpp)
target_link_libraries(test_devices vulkan_interpolator Eigen3::Eigen tbb)

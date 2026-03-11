cmake_minimum_required(VERSION 3.16)

project(arducam_uvc_stereo_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ArducamUvcStereoSdk CONFIG REQUIRED)

add_executable(read_calib_data read_calib_data.cpp)
target_link_libraries(read_calib_data PRIVATE ArducamUvcStereoSdk::cpp_shared)

add_executable(write_calib_data write_calib_data.cpp)
target_link_libraries(write_calib_data PRIVATE ArducamUvcStereoSdk::cpp_shared)

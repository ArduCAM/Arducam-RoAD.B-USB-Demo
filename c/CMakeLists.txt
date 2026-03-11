cmake_minimum_required(VERSION 3.16)

project(arducam_uvc_stereo_demo LANGUAGES C)

find_package(ArducamUvcStereoSdk CONFIG REQUIRED)

add_executable(read_calib_data read_calib_data.c)
target_link_libraries(read_calib_data PRIVATE ArducamUvcStereoSdk::c_shared)

add_executable(write_calib_data write_calib_data.c)
target_link_libraries(write_calib_data PRIVATE ArducamUvcStereoSdk::c_shared)

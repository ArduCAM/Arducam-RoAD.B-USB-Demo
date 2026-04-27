cmake_minimum_required(VERSION 3.8)
project(arducam_uvc_stereo_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# OpenCV discovery:
# - By default use pkg-config to avoid broken OpenCVConfig.cmake installs that
#   reference missing .so paths on some systems.
# - If you have a working OpenCVConfig.cmake, you can enable config mode via
#   -DARDUCAM_USE_OPENCV_CONFIG=ON (and optionally -DOpenCV_DIR=...).
set(OpenCV_FOUND FALSE)
option(ARDUCAM_USE_OPENCV_CONFIG "Use find_package(OpenCV) config mode" OFF)
if(ARDUCAM_USE_OPENCV_CONFIG)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio calib3d)
endif()
if(NOT OpenCV_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENCV4 REQUIRED opencv4)

  set(OPENCV4_EXISTING_INCLUDE_DIRS "")
  foreach(_dir IN LISTS OPENCV4_INCLUDE_DIRS)
    if(IS_DIRECTORY "${_dir}")
      list(APPEND OPENCV4_EXISTING_INCLUDE_DIRS "${_dir}")
    endif()
  endforeach()
  if(OPENCV4_EXISTING_INCLUDE_DIRS STREQUAL "" AND IS_DIRECTORY "/usr/include/opencv4")
    list(APPEND OPENCV4_EXISTING_INCLUDE_DIRS "/usr/include/opencv4")
  endif()

  set(OPENCV4_EXISTING_LIBRARY_DIRS "")
  foreach(_dir IN LISTS OPENCV4_LIBRARY_DIRS)
    if(IS_DIRECTORY "${_dir}")
      list(APPEND OPENCV4_EXISTING_LIBRARY_DIRS "${_dir}")
    endif()
  endforeach()
  if(OPENCV4_EXISTING_LIBRARY_DIRS STREQUAL "" AND CMAKE_LIBRARY_ARCHITECTURE AND IS_DIRECTORY "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    list(APPEND OPENCV4_EXISTING_LIBRARY_DIRS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
  endif()
  if(IS_DIRECTORY "/usr/lib/aarch64-linux-gnu")
    list(APPEND OPENCV4_EXISTING_LIBRARY_DIRS "/usr/lib/aarch64-linux-gnu")
  endif()
  if(IS_DIRECTORY "/lib/aarch64-linux-gnu")
    list(APPEND OPENCV4_EXISTING_LIBRARY_DIRS "/lib/aarch64-linux-gnu")
  endif()

  # Some Jetson/Ubuntu images ship OpenCV libraries with a ".so.<ver>d" suffix
  # but without the unversioned "libopencv_*.so" symlinks required for -lopencv_*.
  # Resolve the exact library files we need without hard-coding absolute filenames.
  set(_opencv_lib_search_paths ${OPENCV4_EXISTING_LIBRARY_DIRS} "/usr/local/lib" "/usr/lib")
  foreach(_component IN ITEMS core imgproc videoio calib3d)
    find_library(OPENCV4_LIB_${_component}
      NAMES opencv_${_component}
      PATHS ${_opencv_lib_search_paths}
      NO_CACHE
    )
    if(NOT OPENCV4_LIB_${_component})
      # Fallback: match versioned library file directly (no symlink case).
      find_file(OPENCV4_FILE_${_component}
        NAMES
          libopencv_${_component}.so.4.5.4d
          libopencv_${_component}.so.4.5.3d
          libopencv_${_component}.so.4.5.2d
          libopencv_${_component}.so.4.5.1d
          libopencv_${_component}.so.4.5.0d
          libopencv_${_component}.so.4.5d
          libopencv_${_component}.so.4.5.4
          libopencv_${_component}.so.4.5
          libopencv_${_component}.so.4
        PATHS ${_opencv_lib_search_paths}
        NO_CACHE
      )
      if(OPENCV4_FILE_${_component})
        set(OPENCV4_LIB_${_component} "${OPENCV4_FILE_${_component}}")
      else()
        message(FATAL_ERROR "Failed to locate OpenCV library for component '${_component}'. Searched: ${_opencv_lib_search_paths}")
      endif()
    endif()
  endforeach()
endif()

add_executable(stereo_camera_cpp_node src/stereo_camera_node.cpp)
target_compile_features(stereo_camera_cpp_node PUBLIC cxx_std_17)
if(OpenCV_FOUND)
  target_include_directories(stereo_camera_cpp_node PRIVATE ${OpenCV_INCLUDE_DIRS})
else()
  target_include_directories(stereo_camera_cpp_node PRIVATE ${OPENCV4_EXISTING_INCLUDE_DIRS})
endif()
ament_target_dependencies(stereo_camera_cpp_node rclcpp sensor_msgs)
if(OpenCV_FOUND)
  target_link_libraries(stereo_camera_cpp_node ${OpenCV_LIBRARIES})
else()
  target_link_libraries(stereo_camera_cpp_node
    ${OPENCV4_LIB_core}
    ${OPENCV4_LIB_imgproc}
    ${OPENCV4_LIB_videoio}
    ${OPENCV4_LIB_calib3d}
  )
endif()

install(TARGETS stereo_camera_cpp_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/stereo_camera_node
  scripts/list_devices
  DESTINATION lib/${PROJECT_NAME}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
  )
endif()

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

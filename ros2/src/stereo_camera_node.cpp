#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

namespace
{
struct CameraCalibration
{
  std::string name;
  int width = 0;
  int height = 0;
  cv::Mat k;
  cv::Mat raw_d;
  std::vector<double> publish_d;
  std::string distortion_model;
};

struct StereoCalibration
{
  int version = 0;
  CameraCalibration left;
  CameraCalibration right;
  cv::Mat rotation;
  cv::Mat translation_m;
};

struct CameraInfoPair
{
  sensor_msgs::msg::CameraInfo left;
  sensor_msgs::msg::CameraInfo right;
  int width = 0;
  int height = 0;
};

class CalibrationError : public std::runtime_error
{
public:
  explicit CalibrationError(const std::string & message) : std::runtime_error(message) {}
};

class FrameFormatError : public std::runtime_error
{
public:
  explicit FrameFormatError(const std::string & message) : std::runtime_error(message) {}
};

std::string fourcc_to_string(int value)
{
  std::string text;
  for (int i = 0; i < 4; ++i) {
    const char c = static_cast<char>((value >> (8 * i)) & 0xFF);
    if (c != '\0') {
      text.push_back(c);
    }
  }
  return text.empty() ? "----" : text;
}

int fourcc_from_string(const std::string & value)
{
  if (value.size() != 4) {
    throw std::invalid_argument("pixel_format must be exactly four characters, got " + value);
  }
  return cv::VideoWriter::fourcc(value[0], value[1], value[2], value[3]);
}

cv::Mat parse_matrix(const cv::FileNode & node, int rows, int cols, const std::string & field_name)
{
  if (node.empty() || !node.isSeq()) {
    throw CalibrationError(field_name + " must be a sequence.");
  }

  cv::Mat matrix(rows, cols, CV_64F);
  if (node.size() == static_cast<size_t>(rows)) {
    for (int r = 0; r < rows; ++r) {
      cv::FileNode row = node[r];
      if (!row.isSeq() || row.size() != static_cast<size_t>(cols)) {
        throw CalibrationError(field_name + " must have shape " + std::to_string(rows) + "x" + std::to_string(cols) + ".");
      }
      for (int c = 0; c < cols; ++c) {
        matrix.at<double>(r, c) = static_cast<double>(row[c]);
      }
    }
    return matrix;
  }

  if (node.size() == static_cast<size_t>(rows * cols)) {
    for (int i = 0; i < rows * cols; ++i) {
      matrix.at<double>(i / cols, i % cols) = static_cast<double>(node[i]);
    }
    return matrix;
  }

  throw CalibrationError(field_name + " must have shape " + std::to_string(rows) + "x" + std::to_string(cols) + ".");
}

cv::Mat parse_vector(const cv::FileNode & node, int count, const std::string & field_name)
{
  if (node.empty() || !node.isSeq() || node.size() != static_cast<size_t>(count)) {
    throw CalibrationError(field_name + " must contain exactly " + std::to_string(count) + " values.");
  }

  cv::Mat vector(count, 1, CV_64F);
  for (int i = 0; i < count; ++i) {
    vector.at<double>(i, 0) = static_cast<double>(node[i]);
  }
  return vector;
}

cv::Mat parse_distortion(const cv::FileNode & node, const std::string & field_name)
{
  if (node.empty() || !node.isSeq() || node.size() < 5) {
    throw CalibrationError(field_name + " must contain at least 5 values.");
  }

  cv::Mat distortion(1, static_cast<int>(node.size()), CV_64F);
  for (int i = 0; i < static_cast<int>(node.size()); ++i) {
    distortion.at<double>(0, i) = static_cast<double>(node[i]);
  }
  return distortion;
}

std::vector<double> publish_distortion(const cv::Mat & raw_distortion, std::string & distortion_model)
{
  const int count = raw_distortion.cols >= 8 ? 8 : 5;
  distortion_model = count == 8 ? "rational_polynomial" : "plumb_bob";

  std::vector<double> values;
  values.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    values.push_back(raw_distortion.at<double>(0, i));
  }
  return values;
}

CameraCalibration parse_camera_entry(const std::string & name, const cv::FileNode & entry)
{
  CameraCalibration camera;
  camera.name = name;
  camera.width = static_cast<int>(entry["width"]);
  camera.height = static_cast<int>(entry["height"]);
  if (camera.width <= 0 || camera.height <= 0) {
    throw CalibrationError(name + " camera width/height must be positive.");
  }

  camera.k = parse_matrix(entry["intrinsicMatrix"], 3, 3, name + ".intrinsicMatrix");
  camera.raw_d = parse_distortion(entry["dist_coeff"], name + ".dist_coeff");
  camera.publish_d = publish_distortion(camera.raw_d, camera.distortion_model);
  return camera;
}

[[maybe_unused]] StereoCalibration parse_stereo_calibration(
  const std::string & json_path,
  int version,
  double translation_scale_to_meter)
{
  cv::FileStorage storage(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (!storage.isOpened()) {
    throw CalibrationError("Failed to open calibration JSON: " + json_path);
  }

  cv::FileNode camera_data = storage["cameraData"];
  if (camera_data.empty() || !camera_data.isSeq()) {
    throw CalibrationError("Calibration JSON must contain a cameraData list.");
  }

  cv::FileNode left_node;
  cv::FileNode right_node;
  for (auto it = camera_data.begin(); it != camera_data.end(); ++it) {
    const std::string name = static_cast<std::string>((*it)["name"]);
    if (name == "left") {
      left_node = *it;
    } else if (name == "right") {
      right_node = *it;
    }
  }

  if (left_node.empty() || right_node.empty()) {
    throw CalibrationError("Calibration JSON must contain exactly one left and one right camera entry.");
  }

  StereoCalibration calibration;
  calibration.version = version;
  calibration.left = parse_camera_entry("left", left_node);
  calibration.right = parse_camera_entry("right", right_node);

  if (calibration.left.width != calibration.right.width || calibration.left.height != calibration.right.height) {
    throw CalibrationError("Left and right calibration dimensions must match.");
  }

  cv::FileNode extrinsics = left_node["extrinsics"];
  if (extrinsics.empty() || !extrinsics.isMap()) {
    throw CalibrationError("Left camera calibration must contain extrinsics to the right camera.");
  }
  const std::string to_cam = static_cast<std::string>(extrinsics["to_cam"]);
  if (to_cam != "right") {
    throw CalibrationError("Left camera extrinsics must point to the right camera.");
  }

  calibration.rotation = parse_matrix(extrinsics["rotationMatrix"], 3, 3, "extrinsics.rotationMatrix");
  calibration.translation_m = parse_vector(extrinsics["translation"], 3, "extrinsics.translation") * translation_scale_to_meter;
  return calibration;
}

cv::Mat scale_intrinsic_matrix(const cv::Mat & matrix, double scale_x, double scale_y)
{
  cv::Mat scaled = matrix.clone();
  scaled.at<double>(0, 0) *= scale_x;
  scaled.at<double>(1, 1) *= scale_y;
  scaled.at<double>(0, 2) *= scale_x;
  scaled.at<double>(1, 2) *= scale_y;
  return scaled;
}

template<std::size_t N>
void fill_array_from_mat(std::array<double, N> & target, const cv::Mat & matrix)
{
  const cv::Mat flat = matrix.reshape(1, 1);
  if (flat.cols != static_cast<int>(N)) {
    throw CalibrationError("Unexpected matrix size while building CameraInfo.");
  }
  for (std::size_t i = 0; i < N; ++i) {
    target[i] = flat.at<double>(0, static_cast<int>(i));
  }
}

[[maybe_unused]] CameraInfoPair build_camera_info_pair(
  const StereoCalibration & calibration,
  int actual_width,
  int actual_height,
  const std::string & left_frame_id,
  const std::string & right_frame_id)
{
  if (actual_width <= 0 || actual_height <= 0) {
    throw CalibrationError("Output image size must be positive.");
  }

  const double scale_x = static_cast<double>(actual_width) / static_cast<double>(calibration.left.width);
  const double scale_y = static_cast<double>(actual_height) / static_cast<double>(calibration.left.height);
  if (std::fabs(scale_x - scale_y) > std::max(1e-9, 1e-6 * std::fabs(scale_y))) {
    throw CalibrationError(
      "Output size " + std::to_string(actual_width) + "x" + std::to_string(actual_height) +
      " changes aspect ratio relative to calibration " + std::to_string(calibration.left.width) +
      "x" + std::to_string(calibration.left.height) + ".");
  }

  cv::Mat left_k = scale_intrinsic_matrix(calibration.left.k, scale_x, scale_y);
  cv::Mat right_k = scale_intrinsic_matrix(calibration.right.k, scale_x, scale_y);
  cv::Mat r1;
  cv::Mat r2;
  cv::Mat p1;
  cv::Mat p2;
  cv::Mat q;
  cv::stereoRectify(
    left_k,
    calibration.left.raw_d,
    right_k,
    calibration.right.raw_d,
    cv::Size(actual_width, actual_height),
    calibration.rotation,
    calibration.translation_m,
    r1,
    r2,
    p1,
    p2,
    q,
    cv::CALIB_ZERO_DISPARITY,
    0.0);

  CameraInfoPair pair;
  pair.width = actual_width;
  pair.height = actual_height;

  pair.left.header.frame_id = left_frame_id;
  pair.left.width = static_cast<uint32_t>(actual_width);
  pair.left.height = static_cast<uint32_t>(actual_height);
  pair.left.distortion_model = calibration.left.distortion_model;
  pair.left.d = calibration.left.publish_d;
  fill_array_from_mat(pair.left.k, left_k);
  fill_array_from_mat(pair.left.r, r1);
  fill_array_from_mat(pair.left.p, p1);

  pair.right.header.frame_id = right_frame_id;
  pair.right.width = static_cast<uint32_t>(actual_width);
  pair.right.height = static_cast<uint32_t>(actual_height);
  pair.right.distortion_model = calibration.right.distortion_model;
  pair.right.d = calibration.right.publish_d;
  fill_array_from_mat(pair.right.k, right_k);
  fill_array_from_mat(pair.right.r, r2);
  fill_array_from_mat(pair.right.p, p2);

  return pair;
}

struct SplitFrame
{
  cv::Mat left;
  cv::Mat right;
  std::string encoding;
};

SplitFrame split_stereo_frame(const cv::Mat & frame)
{
  if (frame.empty()) {
    throw FrameFormatError("Capture returned an empty frame.");
  }
  if (frame.dims != 2) {
    throw FrameFormatError("Unsupported frame rank; expected mono or BGR image.");
  }
  if (frame.cols % 2 != 0) {
    throw FrameFormatError("Combined stereo frame width must be even, got " + std::to_string(frame.cols) + ".");
  }

  const int half_width = frame.cols / 2;
  const cv::Rect left_rect(0, 0, half_width, frame.rows);
  const cv::Rect right_rect(half_width, 0, half_width, frame.rows);

  SplitFrame split;
  if (frame.channels() == 1) {
    split.encoding = sensor_msgs::image_encodings::MONO8;
  } else if (frame.channels() == 3) {
    split.encoding = sensor_msgs::image_encodings::BGR8;
  } else {
    throw FrameFormatError("Unsupported channel count " + std::to_string(frame.channels()) + "; only mono8 and bgr8 are supported.");
  }

  split.left = frame(left_rect);
  split.right = frame(right_rect);
  return split;
}

rclcpp::QoS build_publisher_qos(const std::string & reliability_mode)
{
  rclcpp::QoS qos(rclcpp::SensorDataQoS().keep_last(5));
  if (reliability_mode == "best_effort") {
    qos.best_effort();
  } else if (reliability_mode == "reliable") {
    qos.reliable();
  } else {
    throw std::invalid_argument("Unsupported qos_reliability '" + reliability_mode + "'. Expected 'best_effort' or 'reliable'.");
  }
  qos.durability_volatile();
  return qos;
}

std::vector<double> declare_double_array_parameter(
  rclcpp::Node & node,
  const std::string & name,
  const std::vector<double> & default_value)
{
  return node.declare_parameter<std::vector<double>>(name, default_value);
}

template<std::size_t N>
void copy_vector_to_array(
  const std::vector<double> & source,
  std::array<double, N> & target,
  const std::string & name)
{
  if (source.size() != N) {
    throw CalibrationError(name + " must contain exactly " + std::to_string(N) + " values.");
  }
  for (std::size_t i = 0; i < N; ++i) {
    target[i] = source[i];
  }
}

sensor_msgs::msg::CameraInfo build_camera_info_from_parameters(
  rclcpp::Node & node,
  const std::string & prefix,
  const std::string & frame_id)
{
  sensor_msgs::msg::CameraInfo info;
  info.header.frame_id = frame_id;
  info.width = static_cast<uint32_t>(node.declare_parameter<int>(prefix + "_camera_info_width", 0));
  info.height = static_cast<uint32_t>(node.declare_parameter<int>(prefix + "_camera_info_height", 0));
  info.distortion_model = node.declare_parameter<std::string>(prefix + "_camera_info_distortion_model", "");
  info.d = declare_double_array_parameter(node, prefix + "_camera_info_d", {});
  copy_vector_to_array(
    declare_double_array_parameter(node, prefix + "_camera_info_k", std::vector<double>(9, 0.0)),
    info.k,
    prefix + "_camera_info_k");
  copy_vector_to_array(
    declare_double_array_parameter(node, prefix + "_camera_info_r", std::vector<double>(9, 0.0)),
    info.r,
    prefix + "_camera_info_r");
  copy_vector_to_array(
    declare_double_array_parameter(node, prefix + "_camera_info_p", std::vector<double>(12, 0.0)),
    info.p,
    prefix + "_camera_info_p");
  if (info.width == 0 || info.height == 0) {
    throw CalibrationError(prefix + " camera_info width/height must be provided by the Python bootstrap.");
  }
  if (info.distortion_model.empty()) {
    throw CalibrationError(prefix + " camera_info distortion model must be provided by the Python bootstrap.");
  }
  return info;
}

}  // namespace

class StereoCameraNode : public rclcpp::Node
{
public:
  explicit StereoCameraNode(const rclcpp::NodeOptions & options)
  : Node("stereo_camera_node", options)
  {
    declare_parameter<std::string>("camera_name", "stereo");
    declare_parameter<std::string>("video_node", "");
    declare_parameter<bool>("use_capture_index", false);
    declare_parameter<int>("capture_index", -1);
    declare_parameter<int>("combined_width", 0);
    declare_parameter<int>("combined_height", 0);
    declare_parameter<double>("fps", 30.0);
    declare_parameter<std::string>("pixel_format", "MJPG");
    declare_parameter<std::string>("qos_reliability", "best_effort");
    declare_parameter<double>("translation_scale_to_meter", 0.01);
    declare_parameter<double>("post_calibration_open_delay_sec", 3.0);
    declare_parameter<double>("retry_open_interval_sec", 1.0);
    declare_parameter<std::string>("left_frame_id", "");
    declare_parameter<std::string>("right_frame_id", "");
    declare_parameter<int>("calibration_version", 0);
    declare_parameter<std::string>("capture_source_description", "");

    camera_name_ = get_parameter("camera_name").as_string();
    left_frame_id_ = get_parameter("left_frame_id").as_string();
    right_frame_id_ = get_parameter("right_frame_id").as_string();
    if (left_frame_id_.empty()) {
      left_frame_id_ = camera_name_ + "_left_optical_frame";
    }
    if (right_frame_id_.empty()) {
      right_frame_id_ = camera_name_ + "_right_optical_frame";
    }

    video_node_ = get_parameter("video_node").as_string();
    use_capture_index_ = get_parameter("use_capture_index").as_bool();
    capture_index_ = static_cast<int>(get_parameter("capture_index").as_int());
    requested_combined_width_ = static_cast<int>(get_parameter("combined_width").as_int());
    requested_combined_height_ = static_cast<int>(get_parameter("combined_height").as_int());
    fps_ = get_parameter("fps").as_double();
    pixel_format_ = get_parameter("pixel_format").as_string();
    std::transform(pixel_format_.begin(), pixel_format_.end(), pixel_format_.begin(), [](unsigned char c) {return static_cast<char>(std::toupper(c));});
    retry_open_interval_sec_ = get_parameter("retry_open_interval_sec").as_double();
    capture_source_description_ = get_parameter("capture_source_description").as_string();
    const double post_calibration_open_delay_sec = get_parameter("post_calibration_open_delay_sec").as_double();
    const int calibration_version = static_cast<int>(get_parameter("calibration_version").as_int());

    if (!use_capture_index_ && video_node_.empty()) {
      throw std::runtime_error("video_node is required when use_capture_index is false.");
    }
    if (pixel_format_ != "MJPG" && pixel_format_ != "YUYV") {
      throw std::runtime_error("Unsupported pixel_format '" + pixel_format_ + "'. Expected MJPG or YUYV.");
    }

    camera_info_pair_.left = build_camera_info_from_parameters(*this, "left", left_frame_id_);
    camera_info_pair_.right = build_camera_info_from_parameters(*this, "right", right_frame_id_);
    camera_info_pair_.width = static_cast<int>(camera_info_pair_.left.width);
    camera_info_pair_.height = static_cast<int>(camera_info_pair_.left.height);
    calibration_version_ = calibration_version;

    if (post_calibration_open_delay_sec > 0.0) {
      RCLCPP_INFO(get_logger(), "Waiting %.1fs after calibration read before opening UVC stream.", post_calibration_open_delay_sec);
      std::this_thread::sleep_for(std::chrono::duration<double>(post_calibration_open_delay_sec));
    }

    const auto qos = build_publisher_qos(get_parameter("qos_reliability").as_string());
    left_image_pub_ = create_publisher<sensor_msgs::msg::Image>("left/image_raw", qos);
    right_image_pub_ = create_publisher<sensor_msgs::msg::Image>("right/image_raw", qos);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("left/camera_info", qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("right/camera_info", qos);

    open_capture(true);
    capture_thread_ = std::thread([this]() { capture_loop(); });
  }

  ~StereoCameraNode() override
  {
    stop_.store(true);
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    std::lock_guard<std::mutex> lock(capture_mutex_);
    capture_.release();
  }

private:
  void open_capture(bool initial)
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    capture_.release();

    const int combined_width = requested_combined_width_ > 0 ? requested_combined_width_ : 2 * camera_info_pair_.width;
    const int combined_height = requested_combined_height_ > 0 ? requested_combined_height_ : camera_info_pair_.height;

    RCLCPP_INFO(get_logger(), "Opening UVC capture source %s", source_description().c_str());
    if (use_capture_index_) {
      capture_.open(capture_index_, cv::CAP_V4L2);
    } else {
      capture_.open(video_node_, cv::CAP_V4L2);
    }
    RCLCPP_INFO(get_logger(), "OpenCV VideoCapture open returned; isOpened=%s", capture_.isOpened() ? "true" : "false");

    if (!capture_.isOpened()) {
      throw std::runtime_error("Failed to open UVC capture source " + source_description() + ".");
    }

    RCLCPP_INFO(get_logger(), "Configuring capture requested=%dx%d@%.2f fourcc=%s", combined_width, combined_height, fps_, pixel_format_.c_str());
    if (combined_width > 0) {
      capture_.set(cv::CAP_PROP_FRAME_WIDTH, combined_width);
    }
    if (combined_height > 0) {
      capture_.set(cv::CAP_PROP_FRAME_HEIGHT, combined_height);
    }
    if (fps_ > 0.0) {
      capture_.set(cv::CAP_PROP_FPS, fps_);
    }
    capture_.set(cv::CAP_PROP_FOURCC, fourcc_from_string(pixel_format_));

    RCLCPP_INFO(get_logger(), "Reading initial frame from %s", source_description().c_str());
    cv::Mat frame = read_initial_frame_locked();
    RCLCPP_INFO(get_logger(), "Initial frame received: %dx%d channels=%d", frame.cols, frame.rows, frame.channels());
    RCLCPP_INFO(get_logger(), "Splitting initial stereo frame.");
    SplitFrame split = split_stereo_frame(frame);
    RCLCPP_INFO(
      get_logger(), "Split complete: left=%dx%d right=%dx%d encoding=%s",
      split.left.cols, split.left.rows, split.right.cols, split.right.rows, split.encoding.c_str());
    if (split.left.cols != camera_info_pair_.width || split.left.rows != camera_info_pair_.height) {
      throw CalibrationError(
        "Captured half-frame size " + std::to_string(split.left.cols) + "x" + std::to_string(split.left.rows) +
        " does not match prepared CameraInfo size " + std::to_string(camera_info_pair_.width) + "x" +
        std::to_string(camera_info_pair_.height) + ".");
    }
    RCLCPP_INFO(get_logger(), "CameraInfo pair already prepared by Python bootstrap.");

    const auto previous_width = camera_info_pair_.width;
    const auto previous_height = camera_info_pair_.height;

    const int actual_width = static_cast<int>(std::lround(capture_.get(cv::CAP_PROP_FRAME_WIDTH)));
    const int actual_height = static_cast<int>(std::lround(capture_.get(cv::CAP_PROP_FRAME_HEIGHT)));
    const double actual_fps = capture_.get(cv::CAP_PROP_FPS);
    const std::string fourcc = fourcc_to_string(static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC)));

    if (initial) {
      RCLCPP_INFO(
        get_logger(),
        "Opened %s requested=%dx%d@%.2f actual=%dx%d@%.2f fourcc=%s",
        source_description().c_str(), combined_width, combined_height, fps_, actual_width, actual_height, actual_fps, fourcc.c_str());
      RCLCPP_INFO(
        get_logger(),
        "Calibration version=%d publish_size=%dx%d capture_source=%s",
        calibration_version_, camera_info_pair_.width, camera_info_pair_.height, source_description().c_str());
    } else if (previous_width != camera_info_pair_.width || previous_height != camera_info_pair_.height) {
      RCLCPP_WARN(
        get_logger(),
        "Capture publish size changed from %dx%d to %dx%d after reopen.",
        previous_width, previous_height, camera_info_pair_.width, camera_info_pair_.height);
    }
  }

  cv::Mat read_initial_frame_locked()
  {
    for (int i = 0; i < 20 && !stop_.load(); ++i) {
      cv::Mat frame;
      if (capture_.read(frame) && !frame.empty()) {
        return frame;
      }
      std::this_thread::sleep_for(50ms);
    }
    capture_.release();
    throw std::runtime_error("Timed out waiting for a valid frame from " + source_description() + ".");
  }

  void capture_loop()
  {
    while (rclcpp::ok() && !stop_.load()) {
      try {
        cv::Mat frame;
        {
          std::lock_guard<std::mutex> lock(capture_mutex_);
          if (!capture_.isOpened()) {
            throw std::runtime_error("Capture is not open.");
          }
          if (!capture_.read(frame) || frame.empty()) {
            capture_.release();
            throw std::runtime_error("Frame read failed.");
          }
        }
        SplitFrame split = split_stereo_frame(frame);
        publish_frame_pair(split);
      } catch (const FrameFormatError & exc) {
        RCLCPP_ERROR(get_logger(), "Invalid stereo frame format: %s", exc.what());
        rclcpp::shutdown();
        return;
      } catch (const CalibrationError & exc) {
        RCLCPP_ERROR(get_logger(), "Calibration became incompatible with capture output: %s", exc.what());
        rclcpp::shutdown();
        return;
      } catch (const std::exception & exc) {
        RCLCPP_WARN(get_logger(), "Capture error: %s; reopening stream.", exc.what());
        reopen_capture_loop();
      }
    }
  }

  void reopen_capture_loop()
  {
    while (rclcpp::ok() && !stop_.load()) {
      std::this_thread::sleep_for(std::chrono::duration<double>(retry_open_interval_sec_));
      try {
        open_capture(false);
        RCLCPP_INFO(get_logger(), "Reopened capture stream.");
        return;
      } catch (const CalibrationError & exc) {
        RCLCPP_ERROR(get_logger(), "Reopened capture produced incompatible output: %s", exc.what());
        rclcpp::shutdown();
        return;
      } catch (const FrameFormatError & exc) {
        RCLCPP_ERROR(get_logger(), "Reopened capture produced incompatible output: %s", exc.what());
        rclcpp::shutdown();
        return;
      } catch (const std::exception & exc) {
        RCLCPP_WARN(
          get_logger(), "Failed to reopen %s: %s. Retrying in %.1fs.",
          source_description().c_str(), exc.what(), retry_open_interval_sec_);
      }
    }
  }

  sensor_msgs::msg::Image mat_to_image(
    const cv::Mat & image,
    const std::string & encoding,
    const std::string & frame_id,
    const rclcpp::Time & stamp) const
  {
    sensor_msgs::msg::Image message;
    message.header.stamp = stamp;
    message.header.frame_id = frame_id;
    message.height = static_cast<uint32_t>(image.rows);
    message.width = static_cast<uint32_t>(image.cols);
    message.encoding = encoding;
    message.is_bigendian = false;
    message.step = static_cast<uint32_t>(image.cols * image.elemSize());
    const size_t size = static_cast<size_t>(message.step) * static_cast<size_t>(image.rows);
    message.data.resize(size);
    if (image.isContinuous()) {
      std::memcpy(message.data.data(), image.data, size);
    } else {
      const size_t row_size = static_cast<size_t>(message.step);
      for (int row = 0; row < image.rows; ++row) {
        std::memcpy(message.data.data() + row_size * static_cast<size_t>(row), image.ptr(row), row_size);
      }
    }
    return message;
  }

  void publish_frame_pair(const SplitFrame & split)
  {
    const rclcpp::Time stamp = now();

    auto left_image = mat_to_image(split.left, split.encoding, left_frame_id_, stamp);
    auto right_image = mat_to_image(split.right, split.encoding, right_frame_id_, stamp);

    auto left_info = camera_info_pair_.left;
    auto right_info = camera_info_pair_.right;
    left_info.header.stamp = stamp;
    right_info.header.stamp = stamp;

    left_image_pub_->publish(left_image);
    right_image_pub_->publish(right_image);
    left_info_pub_->publish(left_info);
    right_info_pub_->publish(right_info);
  }

  std::string source_description() const
  {
    if (!capture_source_description_.empty()) {
      return capture_source_description_;
    }
    if (use_capture_index_) {
      return "OpenCV index " + std::to_string(capture_index_);
    }
    return video_node_;
  }

  std::string camera_name_;
  std::string video_node_;
  bool use_capture_index_ = false;
  int capture_index_ = -1;
  int requested_combined_width_ = 0;
  int requested_combined_height_ = 0;
  double fps_ = 30.0;
  std::string pixel_format_ = "MJPG";
  double retry_open_interval_sec_ = 1.0;
  std::string left_frame_id_;
  std::string right_frame_id_;
  std::string capture_source_description_;

  int calibration_version_ = 0;
  CameraInfoPair camera_info_pair_;
  cv::VideoCapture capture_;
  std::mutex capture_mutex_;
  std::atomic_bool stop_{false};
  std::thread capture_thread_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<StereoCameraNode>(rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(false));
    rclcpp::spin(node);
  } catch (const std::exception & exc) {
    std::cerr << "Failed to start stereo camera C++ node: " << exc.what() << std::endl;
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}

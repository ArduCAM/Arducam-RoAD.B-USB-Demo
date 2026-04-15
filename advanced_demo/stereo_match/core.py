from types import SimpleNamespace

import cv2
import numpy as np

from utils import split_stereo_frame


DEFAULT_SETTINGS = {
    "display_mode": "disparity",
    "downscale": 0.5,
    "min_disparity": 0,
    "num_disparities": 128,
    "block_size": 5,
    "uniqueness_ratio": 10,
    "speckle_window_size": 100,
    "speckle_range": 2,
    "disp12_max_diff": 1,
    "post_filter_mode": "bilateral",
    "smooth_diameter": 7,
    "smooth_sigma_color": 2.0,
    "smooth_sigma_space": 7.0,
    "median_ksize": 5,
    "gaussian_ksize": 5,
    "gaussian_sigma": 1.2,
    "wls_lambda": 100.0,
    "wls_sigma_color": 1.5,
    "confidence_threshold": 0,
    "hole_fill": "off",
    "hole_fill_radius": 9,
    "hole_fill_eps": 1e-3,
}

POST_FILTER_MODES = {"off", "bilateral", "median", "gaussian", "wls"}


def has_wls_support():
    ximgproc = getattr(cv2, "ximgproc", None)
    return bool(ximgproc is not None and hasattr(ximgproc, "createDisparityWLSFilter"))


def make_args(settings=None, device_index=None):
    merged = DEFAULT_SETTINGS.copy()
    if settings:
        merged.update(settings)
    merged["device_index"] = device_index
    return SimpleNamespace(**merged)


def validate_args(args):
    if not 0 < args.downscale <= 1.0:
        raise ValueError("--downscale must be in the range (0, 1]")
    if args.num_disparities <= 0 or args.num_disparities % 16 != 0:
        raise ValueError("--num-disparities must be a positive multiple of 16")
    if args.block_size < 3 or args.block_size % 2 == 0:
        raise ValueError("--block-size must be an odd number >= 3")
    if args.post_filter_mode not in POST_FILTER_MODES:
        raise ValueError(
            "--post-filter-mode must be one of: "
            + ", ".join(sorted(POST_FILTER_MODES))
        )

    for name in (
        "uniqueness_ratio",
        "speckle_window_size",
        "speckle_range",
        "disp12_max_diff",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")

    if args.smooth_diameter < 0:
        raise ValueError("--smooth-diameter must be >= 0")
    for name in ("smooth_sigma_color", "smooth_sigma_space"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.median_ksize not in (3, 5):
        raise ValueError("--median-ksize must be 3 or 5")
    if args.gaussian_ksize < 3 or args.gaussian_ksize % 2 == 0:
        raise ValueError("--gaussian-ksize must be an odd number >= 3")
    if args.gaussian_sigma < 0:
        raise ValueError("--gaussian-sigma must be >= 0")
    if args.wls_lambda < 0:
        raise ValueError("--wls-lambda must be >= 0")
    if args.wls_sigma_color < 0:
        raise ValueError("--wls-sigma-color must be >= 0")
    if not 0 <= args.confidence_threshold <= 100:
        raise ValueError("--confidence-threshold must be in the range [0, 100]")
    if args.hole_fill_radius < 0:
        raise ValueError("--hole-fill-radius must be >= 0")
    if args.hole_fill_eps < 0:
        raise ValueError("--hole-fill-eps must be >= 0")


def compute_process_size(img_size, downscale):
    width, height = img_size
    process_width = max(16, int(round(width * downscale)))
    process_height = max(16, int(round(height * downscale)))
    return process_width, process_height


def validate_process_size(process_size, args, context="downscaled image"):
    process_width, process_height = process_size
    if process_width <= abs(args.min_disparity) + args.num_disparities:
        raise ValueError(
            f"{context} width is too small for the requested disparity range: "
            f"{process_width} <= abs({args.min_disparity}) + {args.num_disparities}"
        )
    if args.block_size >= process_width or args.block_size >= process_height:
        raise ValueError(
            f"{context} size is too small for the requested block size: "
            f"{process_width}x{process_height} with block size {args.block_size}"
        )


def create_matcher(args):
    block_area = args.block_size * args.block_size
    return cv2.StereoSGBM_create(
        minDisparity=args.min_disparity,
        numDisparities=args.num_disparities,
        blockSize=args.block_size,
        P1=8 * block_area,
        P2=32 * block_area,
        disp12MaxDiff=args.disp12_max_diff,
        uniquenessRatio=args.uniqueness_ratio,
        speckleWindowSize=args.speckle_window_size,
        speckleRange=args.speckle_range,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def right_matcher_min_disparity(args):
    return -(args.min_disparity + args.num_disparities) + 1


def needs_right_matcher(args):
    return args.post_filter_mode == "wls" or args.confidence_threshold > 0


def create_right_matcher(args):
    block_area = args.block_size * args.block_size
    return cv2.StereoSGBM_create(
        minDisparity=right_matcher_min_disparity(args),
        numDisparities=args.num_disparities,
        blockSize=args.block_size,
        P1=8 * block_area,
        P2=32 * block_area,
        disp12MaxDiff=1000000,
        uniquenessRatio=0,
        speckleWindowSize=0,
        speckleRange=0,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def create_wls_filter(matcher, args):
    if args.post_filter_mode != "wls":
        return None

    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None:
        raise RuntimeError("OpenCV ximgproc module is not available")

    wls_filter = ximgproc.createDisparityWLSFilter(matcher)
    wls_filter.setLambda(float(args.wls_lambda))
    wls_filter.setSigmaColor(float(args.wls_sigma_color))
    return wls_filter


def build_invalid_disparity_frame(shape):
    height, width = shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "No valid disparity",
        (20, min(height - 20, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def colorize_disparity(disparity, min_disparity):
    valid_mask = disparity > (min_disparity - 1.0)
    if not np.any(valid_mask):
        return build_invalid_disparity_frame(disparity.shape[:2])

    valid_values = disparity[valid_mask]
    disp_min = float(valid_values.min())
    disp_max = float(valid_values.max())

    normalized = np.zeros(disparity.shape, dtype=np.uint8)
    if disp_max > disp_min:
        normalized[valid_mask] = np.clip(
            (valid_values - disp_min) * 255.0 / (disp_max - disp_min),
            0,
            255,
        ).astype(np.uint8)
    else:
        normalized[valid_mask] = 255

    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    color[~valid_mask] = 0
    return color


def disparity_valid_mask(disparity, min_disparity):
    return disparity > (min_disparity - 1.0)


def _filter_valid_disparity(disparity, min_disparity, filter_func):
    valid_mask = disparity > (min_disparity - 1.0)
    if not np.any(valid_mask):
        return disparity

    working = disparity.copy()
    working[~valid_mask] = float(np.median(disparity[valid_mask]))
    filtered = filter_func(working)
    output = disparity.copy()
    output[valid_mask] = filtered[valid_mask]
    return output


def apply_bilateral_filter(disparity, min_disparity, diameter, sigma_color, sigma_space):
    if diameter == 0 or sigma_color == 0 or sigma_space == 0:
        return disparity

    return _filter_valid_disparity(
        disparity,
        min_disparity,
        lambda working: cv2.bilateralFilter(
            working,
            diameter,
            sigma_color,
            sigma_space,
        ),
    )


def smooth_disparity(disparity, min_disparity, diameter, sigma_color, sigma_space):
    return apply_bilateral_filter(
        disparity,
        min_disparity,
        diameter,
        sigma_color,
        sigma_space,
    )


def apply_median_filter(disparity, min_disparity, ksize):
    return _filter_valid_disparity(
        disparity,
        min_disparity,
        lambda working: cv2.medianBlur(working, ksize),
    )


def apply_gaussian_filter(disparity, min_disparity, ksize, sigma):
    return _filter_valid_disparity(
        disparity,
        min_disparity,
        lambda working: cv2.GaussianBlur(working, (ksize, ksize), sigma),
    )


def apply_wls_filter(left_gray, disparity_left_raw, disparity_right_raw, runtime):
    wls_filter = runtime.get("wls_filter")
    if disparity_right_raw is None or wls_filter is None:
        raise RuntimeError("WLS filter runtime is incomplete")

    try:
        filtered = wls_filter.filter(
            disparity_left_raw,
            left_gray,
            None,
            disparity_right_raw,
        )
    except TypeError:
        filtered = wls_filter.filter(disparity_left_raw, left_gray)
    return filtered.astype(np.float32) / 16.0


def nearest_valid_fill(disparity, valid_mask):
    if np.all(valid_mask):
        return disparity

    source = np.where(valid_mask, 0, 255).astype(np.uint8)
    _dist, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    label_values = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    valid_labels = labels[valid_mask]
    label_values[valid_labels] = disparity[valid_mask]

    filled = disparity.copy()
    filled[~valid_mask] = label_values[labels[~valid_mask]]
    return filled


def guided_filter(guide, src, radius, eps):
    window_size = (radius * 2 + 1, radius * 2 + 1)
    mean_guide = cv2.boxFilter(
        guide, cv2.CV_32F, window_size, borderType=cv2.BORDER_REPLICATE
    )
    mean_src = cv2.boxFilter(
        src, cv2.CV_32F, window_size, borderType=cv2.BORDER_REPLICATE
    )
    corr_guide = cv2.boxFilter(
        guide * guide, cv2.CV_32F, window_size, borderType=cv2.BORDER_REPLICATE
    )
    corr_guide_src = cv2.boxFilter(
        guide * src, cv2.CV_32F, window_size, borderType=cv2.BORDER_REPLICATE
    )

    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(
        a, cv2.CV_32F, window_size, borderType=cv2.BORDER_REPLICATE
    )
    mean_b = cv2.boxFilter(
        b, cv2.CV_32F, window_size, borderType=cv2.BORDER_REPLICATE
    )
    return mean_a * guide + mean_b


def fill_disparity_holes(disparity, guide_gray, min_disparity, mode, radius, eps):
    if mode == "off" or radius == 0:
        return disparity

    valid_mask = disparity_valid_mask(disparity, min_disparity)
    if not np.any(valid_mask) or np.all(valid_mask):
        return disparity

    filled = nearest_valid_fill(disparity, valid_mask)
    guide = guide_gray.astype(np.float32) / 255.0
    refined = guided_filter(guide, filled, radius, eps)

    valid_values = disparity[valid_mask]
    refined = np.clip(refined, float(valid_values.min()), float(valid_values.max()))

    output = disparity.copy()
    output[~valid_mask] = refined[~valid_mask]
    return output


def apply_post_filter(disparity, min_disparity, args):
    mode = getattr(args, "post_filter_mode", "bilateral")
    if mode == "off":
        return disparity
    if mode == "bilateral":
        return apply_bilateral_filter(
            disparity,
            min_disparity,
            args.smooth_diameter,
            args.smooth_sigma_color,
            args.smooth_sigma_space,
        )
    if mode == "median":
        return apply_median_filter(
            disparity,
            min_disparity,
            args.median_ksize,
        )
    if mode == "gaussian":
        return apply_gaussian_filter(
            disparity,
            min_disparity,
            args.gaussian_ksize,
            args.gaussian_sigma,
        )
    if mode == "wls":
        return disparity
    raise ValueError(f"unsupported post filter mode: {mode}")


def compute_confidence_map(reference_disparity, disparity_right, args):
    valid_left = disparity_valid_mask(reference_disparity, args.min_disparity)
    if not np.any(valid_left):
        return np.zeros(reference_disparity.shape, dtype=np.float32)

    if disparity_right is None:
        return np.zeros(reference_disparity.shape, dtype=np.float32)

    right_min_disparity = right_matcher_min_disparity(args)
    valid_right = disparity_right > (right_min_disparity - 1.0)

    rows, cols = np.indices(reference_disparity.shape, dtype=np.int32)
    matched_cols = np.rint(cols.astype(np.float32) - reference_disparity).astype(np.int32)
    inside = valid_left & (matched_cols >= 0) & (matched_cols < reference_disparity.shape[1])

    confidence = np.zeros(reference_disparity.shape, dtype=np.float32)
    if not np.any(inside):
        return confidence

    sampled_right = np.zeros(reference_disparity.shape, dtype=np.float32)
    sampled_right[inside] = disparity_right[rows[inside], matched_cols[inside]]

    sampled_valid = np.zeros(reference_disparity.shape, dtype=bool)
    sampled_valid[inside] = valid_right[rows[inside], matched_cols[inside]]

    consistent = inside & sampled_valid
    if not np.any(consistent):
        return confidence

    # Map 0-4 px LR error to a 100-0 confidence score.
    consistency_error = np.abs(
        reference_disparity[consistent] + sampled_right[consistent]
    )
    confidence[consistent] = np.clip(100.0 - 25.0 * consistency_error, 0.0, 100.0)
    return confidence


def apply_confidence_threshold(disparity, reference_disparity, disparity_right, args):
    if args.confidence_threshold <= 0:
        return disparity

    confidence = compute_confidence_map(reference_disparity, disparity_right, args)
    valid_mask = disparity_valid_mask(reference_disparity, args.min_disparity)

    output = disparity.copy()
    output[valid_mask & (confidence < args.confidence_threshold)] = (
        args.min_disparity - 1.0
    )
    return output


def scale_roi(roi, src_size, dst_size):
    src_width, src_height = src_size
    dst_width, dst_height = dst_size
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"invalid source size for ROI scaling: {src_size}")

    x, y, w, h = roi
    scale_x = dst_width / float(src_width)
    scale_y = dst_height / float(src_height)

    left = int(np.floor(x * scale_x))
    top = int(np.floor(y * scale_y))
    right = int(np.ceil((x + w) * scale_x))
    bottom = int(np.ceil((y + h) * scale_y))

    left = min(max(left, 0), dst_width)
    top = min(max(top, 0), dst_height)
    right = min(max(right, left), dst_width)
    bottom = min(max(bottom, top), dst_height)
    return left, top, right - left, bottom - top


def crop_to_roi(image, roi):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError(f"cannot crop empty ROI: {roi}")
    return image[y : y + h, x : x + w]


def roi_size(roi):
    _x, _y, width, height = roi
    return width, height


def compute_display_config(img_size, process_size, overlap_roi, args):
    overlap_roi_proc = scale_roi(overlap_roi, img_size, process_size)
    overlap_size_proc = roi_size(overlap_roi_proc)
    validate_process_size(overlap_size_proc, args, context="cropped valid stereo image")

    disparity_roi_proc = cv2.getValidDisparityROI(
        (0, 0, overlap_size_proc[0], overlap_size_proc[1]),
        (0, 0, overlap_size_proc[0], overlap_size_proc[1]),
        args.min_disparity,
        args.num_disparities,
        args.block_size,
    )
    if disparity_roi_proc[2] <= 0 or disparity_roi_proc[3] <= 0:
        raise ValueError(
            "valid disparity ROI is empty after cropping; try decreasing "
            "--num-disparities or increasing --downscale"
        )

    return {
        "overlap_roi_proc": overlap_roi_proc,
        "disparity_roi_proc": disparity_roi_proc,
    }


def prepare_runtime(img_size, overlap_roi, settings=None, device_index=None):
    args = make_args(settings, device_index=device_index)
    validate_args(args)

    process_size = compute_process_size(img_size, args.downscale)
    validate_process_size(process_size, args)
    display_config = compute_display_config(img_size, process_size, overlap_roi, args)
    matcher = create_matcher(args)
    right_matcher = create_right_matcher(args) if needs_right_matcher(args) else None
    wls_filter = create_wls_filter(matcher, args)

    return {
        "args": args,
        "process_size": process_size,
        "display_config": display_config,
        "matcher": matcher,
        "right_matcher": right_matcher,
        "wls_filter": wls_filter,
    }


def process_disparity_frame(
    frame,
    img_size,
    maps,
    runtime,
    output_rgb=False,
    return_metadata=False,
):
    map_l1, map_l2, map_r1, map_r2 = maps
    args = runtime["args"]
    display_mode = getattr(args, "display_mode", "disparity")
    process_size = runtime["process_size"]
    display_config = runtime["display_config"]

    left, right = split_stereo_frame(frame, img_size)

    output = {
        "display_mode": display_mode,
        "display_config": display_config,
        "process_size": process_size,
        "left_frame": left,
        "right_frame": right,
        "reference_disparity": None,
        "disparity": None,
        "disparity_right": None,
    }
    need_disparity = display_mode == "disparity"

    if not need_disparity:
        if display_mode == "left":
            display_frame = left
        else:
            display_frame = right
        output["display_frame"] = (
            cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB) if output_rgb else display_frame
        )
        return output if return_metadata else output["display_frame"]

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_rect = cv2.remap(left_gray, map_l1, map_l2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_gray, map_r1, map_r2, cv2.INTER_LINEAR)

    if process_size != img_size:
        left_proc = cv2.resize(left_rect, process_size, interpolation=cv2.INTER_AREA)
        right_proc = cv2.resize(right_rect, process_size, interpolation=cv2.INTER_AREA)
    else:
        left_proc = left_rect
        right_proc = right_rect

    overlap_roi_proc = display_config["overlap_roi_proc"]
    disparity_roi_proc = display_config["disparity_roi_proc"]
    left_proc = crop_to_roi(left_proc, overlap_roi_proc)
    right_proc = crop_to_roi(right_proc, overlap_roi_proc)

    disparity_left_raw = runtime["matcher"].compute(left_proc, right_proc)
    reference_disparity = disparity_left_raw.astype(np.float32) / 16.0

    disparity_right_raw = None
    disparity_right = None
    if runtime["right_matcher"] is not None:
        disparity_right_raw = runtime["right_matcher"].compute(right_proc, left_proc)
        disparity_right = disparity_right_raw.astype(np.float32) / 16.0

    if args.post_filter_mode == "wls":
        disparity = apply_wls_filter(
            left_proc,
            disparity_left_raw,
            disparity_right_raw,
            runtime,
        )
    else:
        disparity = reference_disparity
    disparity = apply_confidence_threshold(
        disparity,
        reference_disparity,
        disparity_right,
        args,
    )
    disparity = fill_disparity_holes(
        disparity,
        left_proc,
        args.min_disparity,
        args.hole_fill,
        args.hole_fill_radius,
        args.hole_fill_eps,
    )
    disparity = apply_post_filter(disparity, args.min_disparity, args)
    output["reference_disparity"] = reference_disparity
    output["disparity"] = disparity
    output["disparity_right"] = disparity_right

    disparity_color = colorize_disparity(disparity, args.min_disparity)
    disparity_color = crop_to_roi(disparity_color, disparity_roi_proc)

    if process_size != img_size:
        disparity_color = cv2.resize(
            disparity_color,
            None,
            fx=img_size[0] / float(process_size[0]),
            fy=img_size[1] / float(process_size[1]),
            interpolation=cv2.INTER_NEAREST,
        )

    if display_mode == "disparity":
        display_frame = disparity_color
    elif display_mode == "left":
        display_frame = left
    else:
        display_frame = right

    output["display_frame"] = (
        cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB) if output_rgb else display_frame
    )
    return output if return_metadata else output["display_frame"]


__all__ = [
    "POST_FILTER_MODES",
    "apply_bilateral_filter",
    "apply_confidence_threshold",
    "apply_gaussian_filter",
    "apply_median_filter",
    "apply_post_filter",
    "apply_wls_filter",
    "compute_confidence_map",
    "DEFAULT_SETTINGS",
    "build_invalid_disparity_frame",
    "colorize_disparity",
    "compute_display_config",
    "compute_process_size",
    "create_matcher",
    "create_right_matcher",
    "create_wls_filter",
    "crop_to_roi",
    "disparity_valid_mask",
    "fill_disparity_holes",
    "guided_filter",
    "has_wls_support",
    "make_args",
    "nearest_valid_fill",
    "needs_right_matcher",
    "prepare_runtime",
    "process_disparity_frame",
    "right_matcher_min_disparity",
    "roi_size",
    "scale_roi",
    "smooth_disparity",
    "validate_args",
    "validate_process_size",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import asdict, dataclass

from PIL import Image

DRAFT_INFO_KEY = "vllm_deepseek_v41_draft"


def llm_grid(best_height, best_width, patch_size, downsample_ratio):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in
    max_n_token. Returns (best_height, best_width)."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width,
        math.floor(max_h_float) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    """Shrink the pixel size until the image costs at most max_n_token LLM
    tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def image_geometry(
    width, height, p, downsample_ratio, max_n_token, min_pixels, max_wh_ratio
):
    """Compute the reference V4.1 resize and LLM grid from oriented dimensions."""
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height, width, best_height, best_width, p, downsample_ratio, max_n_token
    )


@dataclass(frozen=True)
class VisionResizeConfig:
    patch_size: int
    downsample_ratio: int
    max_image_tokens: int
    min_pixels: int
    max_wh_ratio: float | None

    def __post_init__(self):
        for name in ("patch_size", "downsample_ratio", "max_image_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_image_tokens < 4:
            raise ValueError("max_image_tokens must be >= 4")
        if type(self.min_pixels) is not int or self.min_pixels < 0:
            raise ValueError("min_pixels must be a nonnegative integer")
        if self.max_wh_ratio is not None and (
            type(self.max_wh_ratio) not in (int, float)
            or not math.isfinite(self.max_wh_ratio)
            or self.max_wh_ratio <= 0
        ):
            raise ValueError("max_wh_ratio must be positive and finite")


def draft_jpeg(image: Image.Image, config: VisionResizeConfig) -> dict | None:
    """Configure JPEG decoding before EXIF normalization or pixel loading.

    Return provenance only when the decoder actually reduces the image.
    """
    if image.format != "JPEG" or image.mode not in ("RGB", "L"):
        return None
    size = image.size
    orientation = image.getexif().get(274, 1)
    oriented = size[::-1] if orientation in (5, 6, 7, 8) else size
    _, _, height, width = image_geometry(
        *oriented,
        config.patch_size,
        config.downsample_ratio,
        config.max_image_tokens,
        config.min_pixels,
        config.max_wh_ratio,
    )
    if min(width, height) <= 0:
        return None
    target = width, height
    decoder_target = target[::-1] if orientation in (5, 6, 7, 8) else target
    if any(src < 2 * dst for src, dst in zip(size, decoder_target)):
        return None
    image.draft(image.mode, decoder_target)
    if image.size == size:
        return None
    return {"original_wh": oriented, "config": asdict(config)}


def pad_draft_image(
    image: Image.Image, target: tuple[int, int], original_wh: tuple[int, int]
) -> Image.Image:
    # Preserve the original aspect ratio even if JPEG draft rounded dimensions.
    width, height = target
    ratio = original_wh[0] / original_wh[1]
    if ratio > width / height:
        contained = (width, round(original_wh[1] / original_wh[0] * width))
    else:
        contained = (round(height * ratio), height)
    if min(contained) == 0:
        return Image.new(image.mode, target, (127, 127, 127))
    resized = image.resize(contained, Image.Resampling.BICUBIC)
    if contained == target:
        return resized
    padded = Image.new(image.mode, target, (127, 127, 127))
    padded.paste(
        resized, (round((width - contained[0]) / 2), round((height - contained[1]) / 2))
    )
    return padded

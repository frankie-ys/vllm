# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal preprocessing for the DeepSeek-V4.1 vision variant.

The image transform and image-span construction are ported from the
official repository's `image_processor.py` so that token counts bit-match
the reference. Each `<｜deepseek_image｜>` placeholder in the prompt expands
to `[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h +
[IMAGE_END]`; every span position carries `image_token_id` (129264) in
`input_ids` and the roles ride along in a per-image `types` tensor
(the reference's out-of-band `token_types`). IMAGE slots receive aligner
rows in reading order; the delimiters take the learned `image_start` /
`image_newline` / `image_end` vectors.

The token stream matches the reference exactly: no compressor-alignment
pad is inserted (the reference pools image tokens across compressor-group
boundaries freely).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import BatchFeature

from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.deepseek_v41_jpeg import (
    DRAFT_INFO_KEY,
    VisionResizeConfig,
    image_geometry,
    num_image_tokens,
    pad_draft_image,
)
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Indexed in numpy: uint8 indices are masks in torch
_NORMALIZE_LUT = (
    ((torch.arange(256, dtype=torch.float32, device="cpu") / 255 - 0.5) / 0.5)
    .to(torch.bfloat16)
    .view(torch.int16)
    .numpy()
)

# The checkpoint tokenizer's id for IMAGE_PLACEHOLDER (the config's
# ``image_token_id``). Every image-span position carries this id, exactly
# like the reference; the MoE router detects image tokens as the range
# [IMAGE_SENTINEL_BASE_ID, IMAGE_SENTINEL_BASE_ID + 5).
IMAGE_SENTINEL_BASE_ID = 129264


def image_sentinel_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Boolean mask for image-span positions."""
    return token_ids == IMAGE_SENTINEL_BASE_ID


def validate_image_sentinel_ids(tokenizer) -> None:
    """Check the image token id against the tokenizer."""
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_id != IMAGE_SENTINEL_BASE_ID:
        raise ValueError(
            f"Image placeholder {IMAGE_PLACEHOLDER!r} has id {image_id}, "
            f"expected {IMAGE_SENTINEL_BASE_ID} (the config's "
            "image_token_id); the DeepSeek-V4.1 vision path keys image "
            "routing and engram masking off this id."
        )


def load_image(
    image: Image.Image,
    *,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio: float | None,
    do_rescale: bool = True,
    do_normalize: bool = True,
):
    """Transform one PIL image into ViT patches.

    Same math as the reference `load_image`, except the image is already
    decoded (vLLM supplies PIL images instead of a record dict).
    """
    p = patch_size
    draft_info = image.info.get(DRAFT_INFO_KEY)
    used_draft = isinstance(draft_info, dict)
    original_wh = image.size
    if used_draft:
        expected_config = VisionResizeConfig(
            patch_size, downsample_ratio, max_n_token, min_pixels, max_wh_ratio
        )
        if draft_info["config"] != asdict(expected_config):
            raise ValueError("JPEG draft config does not match the DeepSeek V4.1 model")
        original_wh = draft_info["original_wh"]
    image = image.convert("RGB")
    width, height = original_wh
    n_llm_h, n_llm_w, best_height, best_width = image_geometry(
        width, height, p, downsample_ratio, max_n_token, min_pixels, max_wh_ratio
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if max_wh_ratio is not None and width >= max_wh_ratio * height:
        image = image.resize((best_width, best_height))
    elif used_draft:
        image = pad_draft_image(image, (best_width, best_height), original_wh)
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    assert do_rescale == do_normalize
    pixels = np.asarray(image, dtype=np.uint8)
    if do_normalize:
        x = torch.from_numpy(_NORMALIZE_LUT[pixels]).view(torch.bfloat16)
    else:
        x = torch.from_numpy(pixels)
    patches = (
        x.permute(2, 0, 1)
        .reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Reading-order span layout: one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


class DeepseekV4VLImageProcessor:
    """Per-image transform (the PIL-input equivalent of the reference
    `load_image`)."""

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.patch_size = config.vision_patch_size
        self.downsample_ratio = config.vision_downsample_ratio
        self.max_n_token = config.vision_max_n_token
        self.min_pixels = config.vision_min_pixels
        self.max_wh_ratio = config.vision_max_wh_ratio

    def __call__(
        self,
        image: Image.Image,
        *,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ):
        return load_image(
            image,
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
            max_n_token=self.max_n_token,
            min_pixels=self.min_pixels,
            max_wh_ratio=self.max_wh_ratio,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
        )


class DeepseekV4VLProcessor:
    """Minimal stand-in for the HF processor of DeepSeek-V4.1 vision models.

    The official repository ships image preprocessing as plain functions in
    `image_processor.py` (no `auto_map` processor), so this class wraps
    their ports directly and the model loads without `--trust-remote-code`.

    `__call__` returns a `BatchFeature` with one entry per image
    (flattened across images):

    - `patches`: `(sum(n_vit_h * n_vit_w), 3, p, p)` bf16 ViT patches.
    - `vit_grid`: `(num_images, 2)` int64 `[n_vit_h, n_vit_w]`.
    - `llm_grid`: `(num_images, 2)` int64 `[n_llm_h, n_llm_w]`.
    - `types`: concatenated per-image pad-free span roles
      (IMAGE_START/IMAGE/IMAGE_NEW_LINE/IMAGE_END); every span position
      carries `image_token_id` in the prompt's token ids.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        *,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.image_processor = DeepseekV4VLImageProcessor(config)
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

    def __call__(
        self,
        text: str | None = None,
        images: Sequence[Image.Image] | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        patches_list = []
        vit_grid = []
        llm_grid_list = []
        types_list = []
        for image in images or []:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.image_processor(
                image, do_rescale=self.do_rescale, do_normalize=self.do_normalize
            )
            patches_list.append(patches)
            vit_grid.append((n_vit_h, n_vit_w))
            llm_grid_list.append((n_llm_h, n_llm_w))
            types_list.append(image_token_types(n_llm_h, n_llm_w))

        if not patches_list:
            return BatchFeature({})

        return BatchFeature(
            {
                "patches": torch.cat(patches_list),
                "vit_grid": torch.tensor(vit_grid, dtype=torch.int64),
                "llm_grid": torch.tensor(llm_grid_list, dtype=torch.int64),
                "types": torch.cat(types_list),
            }
        )


def _get_image_fields(
    inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    vit_grid = inputs.get("vit_grid")
    llm_grid = inputs.get("llm_grid")
    empty = torch.empty(0, dtype=torch.long)
    patch_sizes = empty if vit_grid is None else vit_grid.prod(-1)
    span_sizes = (
        empty if llm_grid is None else llm_grid[:, 0] * (llm_grid[:, 1] + 1) + 2
    )
    return {
        "patches": MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
        "vit_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        "llm_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        "types": MultiModalFieldConfig.flat_from_sizes(
            "image", span_sizes, keep_on_cpu=True
        ),
        "image_embeds": MultiModalFieldConfig.flat_from_sizes("image", span_sizes),
    }


class DeepseekV4VLMultiModalDataParser(MultiModalDataParser):
    embedding_fields = {
        "image": {"image_embeds": "values", "llm_grid": "metadata"},
    }

    def _parse_image_data(self, data):
        if isinstance(data, dict):
            required, optional = self.embedding_field_sets("image")
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields=required,
                optional_fields=optional,
                fields_factory=_get_image_fields,
            )
        return super()._parse_image_data(data)


class DeepseekV4VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> DeepseekV41Config:
        return self.ctx.get_hf_config(DeepseekV41Config)

    def get_hf_processor(  # type: ignore[override]
        self, **kwargs: object
    ) -> DeepseekV4VLProcessor:
        if kwargs:
            raise ValueError(f"Unexpected processor kwargs: {sorted(kwargs)}")

        mm_config = self.ctx.get_mm_config()
        return DeepseekV4VLProcessor(
            self.get_hf_config(),
            do_rescale=not mm_config.mm_device_do_normalize,
            do_normalize=not mm_config.mm_device_do_normalize,
        )

    def get_data_parser(self) -> DeepseekV4VLMultiModalDataParser:
        return DeepseekV4VLMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_hf_config().vision_max_n_token}

    def get_image_placeholder_token_id(self) -> int:
        token_id = self.get_tokenizer().convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if token_id is None:
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        return token_id

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()
        patch_size = hf_config.vision_patch_size
        downsample_ratio = hf_config.vision_downsample_ratio
        budget = hf_config.vision_max_n_token
        max_wh_ratio = hf_config.vision_max_wh_ratio
        best_h = best_w = 0
        # Each LLM row costs an extra newline token. Maximize the ViT grid,
        # including the patch rounding after the optional aspect-ratio clamp.
        for n_llm_h in range(1, (budget - 2) // 2 + 1):
            n_vit_h = n_llm_h * downsample_ratio
            n_vit_w = ((budget - 2) // n_llm_h - 1) * downsample_ratio
            if max_wh_ratio is not None:
                n_vit_w = min(n_vit_w, math.ceil(n_vit_h * max_wh_ratio))
            if n_vit_h * n_vit_w > best_h * best_w:
                best_h, best_w = n_vit_h, n_vit_w
        if best_h == 0:
            raise ValueError(
                "Image token budget and aspect ratio allow no image patches"
            )
        height, width = best_h * patch_size, best_w * patch_size
        # Avoid the min-pixels upscaler perturbing the selected aspect ratio.
        scale = 1
        while height * width * scale * scale < hf_config.vision_min_pixels:
            scale *= 2
        return ImageSize(width=width * scale, height=height * scale)


class DeepseekV4VLDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VLProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
        }


class DeepseekV4VLMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VLProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _get_image_fields(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.get_image_placeholder_token_id()
        validate_image_sentinel_ids(self.info.get_tokenizer())

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            grid: torch.Tensor = out_mm_kwargs["image"][item_idx]["llm_grid"].data
            # Every span position carries image_token_id; the roles live in
            # ``types``. All of them are embed positions (delimiters get the
            # learned vectors from embed_multimodal, not the embed table).
            full = [image_token_id] * num_image_tokens(*grid.tolist())
            return PromptUpdateDetails.select_token_id(full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            ),
        ]

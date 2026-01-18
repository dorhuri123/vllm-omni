# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import PIL.Image
import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest


def get_anisora_i2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    try:
        from diffusers.video_processor import VideoProcessor
    except Exception:
        VideoProcessor = None  # type: ignore

    video_processor = VideoProcessor(vae_scale_factor=8) if VideoProcessor else None

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent" or video_processor is None:
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_anisora_i2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for AniSora I2V: load/resize input image if provided."""
    try:
        import numpy as np  # noqa: F401
    except Exception:
        pass

    def pre_process_func(requests: list[OmniDiffusionRequest]) -> list[OmniDiffusionRequest]:
        for req in requests:
            if req.image_path is not None and req.pil_image is None:
                req.pil_image = PIL.Image.open(req.image_path).convert("RGB")
        return requests

    return pre_process_func


class AniSoraI2VPipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        # Components will be implemented later (tokenizer/encoders/vae/transformers/scheduler)

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | tuple[float, float] = 4.0,
        frame_num: int | None = None,
        output_type: str | None = "np",
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        raise NotImplementedError("AniSoraI2VPipeline forward not implemented yet.")

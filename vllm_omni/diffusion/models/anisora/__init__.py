# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Index-AniSora model support for vLLM-Omni.

AniSora (V1.0) is built on CogVideoX architecture (not Wan).
This module provides Image-to-Video (I2V) generation pipeline.

Model: Disty0/Index-anisora-5B-diffusers
Architecture: CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
"""

from .pipeline_anisora_i2v_cogvideox import (
    AniSoraI2VCogVideoXPipeline,
)

__all__ = [
    "AniSoraI2VCogVideoXPipeline",
]

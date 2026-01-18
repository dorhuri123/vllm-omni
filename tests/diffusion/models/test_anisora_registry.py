# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.registry import (
    DIFFUSION_POST_PROCESS_MAP,
    DIFFUSION_PRE_PROCESS_MAP,
    PIPELINE_REGISTRY,
)


def test_anisora_registry_entries_present():
    assert "AniSoraPipeline" in PIPELINE_REGISTRY
    assert "AniSoraImageToVideoPipeline" in PIPELINE_REGISTRY

    assert "AniSoraPipeline" in DIFFUSION_PRE_PROCESS_MAP
    assert "AniSoraPipeline" in DIFFUSION_POST_PROCESS_MAP

    assert "AniSoraImageToVideoPipeline" in DIFFUSION_PRE_PROCESS_MAP
    assert "AniSoraImageToVideoPipeline" in DIFFUSION_POST_PROCESS_MAP

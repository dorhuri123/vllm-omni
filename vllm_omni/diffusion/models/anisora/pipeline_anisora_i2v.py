# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import PIL.Image
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel

from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest


def get_anisora_i2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_anisora_i2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for AniSora I2V: load input image if provided."""

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
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model) if isinstance(model, str) else False

        # Load configs
        transformer_config = self._load_transformer_config(model, "transformer", local_files_only)

        # Components
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        # Transformer
        self.transformer = self._create_transformer_from_config(transformer_config)

        # Scheduler
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

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
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if req.pil_image is None and req.image_path is None:
            raise ValueError("AniSora I2V requires an input image (path or PIL image).")

        prompt = req.prompt if req.prompt is not None else prompt
        negative_prompt = req.negative_prompt if req.negative_prompt is not None else negative_prompt

        height = req.height or height or 720
        width = req.width or width or 1280
        num_frames = req.num_frames if req.num_frames else frame_num or 81
        num_steps = req.num_inference_steps or num_inference_steps or 40

        # Ensure divisibility
        patch_size = self.transformer.config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
        self._guidance_scale = guidance_low

        self._check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        # Generator
        if generator is None:
            generator = req.generator
        if generator is None and req.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.seed)

        # Encode prompts
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self._encode_prompt(
                prompt=prompt or "",
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_low > 1.0,
                num_videos_per_prompt=req.num_outputs_per_prompt or 1,
                max_sequence_length=req.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # Prepare noise latents using out_channels (condition added separately)
        batch_size = prompt_embeds.shape[0]
        latents = self._prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.transformer.config.out_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=req.latents,
        )

        # Image condition
        from diffusers.video_processor import VideoProcessor

        image = req.pil_image
        if image is None and req.image_path is not None:
            image = PIL.Image.open(req.image_path).convert("RGB")

        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        image_tensor = video_processor.preprocess(image, height=height, width=width)
        image_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=self.vae.dtype)

        latent_condition = self.vae.encode(image_tensor).latent_dist.mode()
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        # Normalize to DiT input space
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_condition.device, latent_condition.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latent_condition.device, latent_condition.dtype
        )
        latent_condition = (latent_condition - latents_mean) * latents_std
        latent_condition = latent_condition.to(torch.float32)

        # Create first-frame mask: 0 for frame 0 (condition), 1 for others
        num_latent_frames = latents.shape[2]
        latent_height = latents.shape[3]
        latent_width = latents.shape[4]
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
        )
        first_frame_mask[:, :, 0] = 0

        if attention_kwargs is None:
            attention_kwargs = {}

        # Denoising loop with condition blending
        for t in timesteps:
            self._current_timestep = t

            latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
            latent_model_input = latent_model_input.to(dtype)
            timestep = t.expand(latents.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            if guidance_low > 1.0 and negative_prompt_embeds is not None:
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_low * (noise_pred - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        self._current_timestep = None

        # Blend final latents with condition for frame 0
        latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        # Decode
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def _check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def _encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @staticmethod
    def _load_transformer_config(
        model_path: str, subfolder: str = "transformer", local_files_only: bool = True
    ) -> dict:
        if local_files_only:
            config_path = os.path.join(model_path, subfolder, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    return json.load(f)
        else:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(repo_id=model_path, filename=f"{subfolder}/config.json")
                with open(config_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def _create_transformer_from_config(config: dict) -> WanTransformer3DModel:
        kwargs = {}
        if "patch_size" in config:
            kwargs["patch_size"] = tuple(config["patch_size"])
        if "num_attention_heads" in config:
            kwargs["num_attention_heads"] = config["num_attention_heads"]
        if "attention_head_dim" in config:
            kwargs["attention_head_dim"] = config["attention_head_dim"]
        if "in_channels" in config:
            kwargs["in_channels"] = config["in_channels"]
        if "out_channels" in config:
            kwargs["out_channels"] = config["out_channels"]
        if "text_dim" in config:
            kwargs["text_dim"] = config["text_dim"]
        if "freq_dim" in config:
            kwargs["freq_dim"] = config["freq_dim"]
        if "ffn_dim" in config:
            kwargs["ffn_dim"] = config["ffn_dim"]
        if "num_layers" in config:
            kwargs["num_layers"] = config["num_layers"]
        if "cross_attn_norm" in config:
            kwargs["cross_attn_norm"] = config["cross_attn_norm"]
        if "eps" in config:
            kwargs["eps"] = config["eps"]
        if "image_dim" in config:
            kwargs["image_dim"] = config["image_dim"]
        if "added_kv_proj_dim" in config:
            kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
        if "rope_max_seq_len" in config:
            kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]
        if "pos_embed_seq_len" in config:
            kwargs["pos_embed_seq_len"] = config["pos_embed_seq_len"]

        return WanTransformer3DModel(**kwargs)

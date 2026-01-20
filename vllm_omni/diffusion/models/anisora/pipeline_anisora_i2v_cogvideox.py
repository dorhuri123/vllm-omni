# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AniSora I2V Pipeline using native CogVideoX components from diffusers.

AniSora is based on CogVideoX architecture, not Wan. This pipeline uses
diffusers' CogVideoXTransformer3DModel and AutoencoderKLCogVideoX directly.
"""

from __future__ import annotations

import os

import PIL.Image
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.utils import get_local_device


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


class AniSoraI2VCogVideoXPipeline(nn.Module):
    """
    AniSora Image-to-Video Pipeline using native CogVideoX architecture.

    This pipeline uses diffusers' native CogVideoX components for compatibility
    with the Disty0/Index-anisora-5B-diffusers model.
    """

    def __init__(
        self,
        *,
        model_path: str = "Disty0/Index-anisora-5B-diffusers",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or get_local_device()
        self.dtype = dtype

        local_files_only = os.path.exists(model_path) if isinstance(model_path, str) else False

        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        print(f"Loading text encoder from {model_path}...")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

        print(f"Loading VAE from {model_path}...")
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE in float32 for precision
            local_files_only=local_files_only,
        )

        print(f"Loading transformer from {model_path}...")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

        print(f"Loading scheduler from {model_path}...")
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Scale factors from VAE config
        self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio

        self._current_timestep = None
        print("Pipeline loaded successfully!")

    def to(self, device):
        """Move pipeline to device."""
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        return self

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 226,
    ):
        """Encode text prompts using T5."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        prompt_embeds = self.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        # Negative prompt
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            uncond_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.dtype, device=self.device)
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def encode_image(self, image: PIL.Image.Image, height: int, width: int):
        """Encode input image to latent space."""
        # Resize image
        image = image.resize((width, height), PIL.Image.LANCZOS)

        # Convert to tensor: [0, 255] -> [-1, 1]
        image_tensor = (
            torch.from_numpy((torch.tensor([list(image.getdata())]).reshape(1, height, width, 3).numpy() / 127.5) - 1.0)
            .permute(0, 3, 1, 2)
            .to(dtype=torch.float32, device=self.device)
        )

        # Add frame dimension: [B, C, H, W] -> [B, C, F, H, W] with F=1
        # CogVideoX VAE expects [B, C, F, H, W]
        image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]

        # Encode
        latent = self.vae.encode(image_tensor).latent_dist.sample()

        # Scale by latent std if available
        if hasattr(self.vae.config, "scaling_factor"):
            latent = latent * self.vae.config.scaling_factor

        return latent  # [B, C, 1, H//8, W//8]

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
    ):
        """Prepare 3D rotary positional embeddings."""
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0 style
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=self.device,
            )
        else:
            # CogVideoX 1.5 style
            base_num_frames = (num_frames + p_t - 1) // p_t
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=self.device,
            )

        return freqs_cos, freqs_sin

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image,
        negative_prompt: str | list[str] | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 17,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        generator: torch.Generator | None = None,
        output_type: str = "tensor",
    ):
        """
        Generate video from image and text prompt.

        Args:
            prompt: Text prompt(s)
            image: Input image
            negative_prompt: Negative prompt(s)
            height: Output height
            width: Output width
            num_frames: Number of output frames
            num_inference_steps: Denoising steps
            guidance_scale: Classifier-free guidance scale
            generator: Random generator for reproducibility
            output_type: "tensor", "np", or "pil"
        """
        # Encode prompt
        print("Encoding prompts...")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt)

        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None

        # Encode image
        print("Encoding image...")
        image_latents = self.encode_image(image, height, width)

        # Prepare latent dimensions
        batch_size = prompt_embeds.shape[0]
        num_channels_latents = self.transformer.config.in_channels // 2  # 16 for noise, 16 for image
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # CogVideoX uses [B, F, C, H, W] format
        latent_shape = (batch_size, latent_num_frames, num_channels_latents, latent_height, latent_width)

        # Initial noise
        print("Preparing latents...")
        latents = randn_tensor(latent_shape, generator=generator, device=self.device, dtype=self.dtype)

        # Prepare image latents for conditioning
        # image_latents: [B, C, 1, H, W] -> [B, 1, C, H, W]
        image_latents = image_latents.permute(0, 2, 1, 3, 4).to(dtype=self.dtype)

        # Pad image latents to match num_frames: first frame is image, rest are zeros
        padding_shape = (batch_size, latent_num_frames - 1, num_channels_latents, latent_height, latent_width)
        latent_padding = torch.zeros(padding_shape, device=self.device, dtype=self.dtype)
        image_latents_padded = torch.cat([image_latents, latent_padding], dim=1)  # [B, F, C, H, W]

        # Prepare rotary embeddings
        print("Preparing rotary embeddings...")
        image_rotary_emb = self._prepare_rotary_positional_embeddings(height, width, latent_num_frames)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Scale initial noise
        latents = latents * self.scheduler.init_noise_sigma

        print(f"Starting denoising loop ({num_inference_steps} steps)...")
        for i, t in enumerate(timesteps):
            self._current_timestep = t

            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Expand image latents for CFG
            latent_image_input = (
                torch.cat([image_latents_padded] * 2) if do_classifier_free_guidance else image_latents_padded
            )

            # Concatenate noise and image latents along channel dimension
            # [B, F, C, H, W] + [B, F, C, H, W] -> [B, F, 2C, H, W]
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

            # Prepare prompt embeds for CFG
            if do_classifier_free_guidance:
                prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                prompt_embeds_input = prompt_embeds

            # Expand timestep to match batch dimension (for CFG)
            batch_size = latent_model_input.shape[0]
            timestep = t.expand(batch_size).to(latent_model_input.dtype)

            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds_input,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            # CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % 10 == 0:
                print(f"  Step {i + 1}/{num_inference_steps}")

        self._current_timestep = None
        print("Decoding latents...")

        # Decode latents
        # CogVideoX VAE expects [B, C, F, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W] -> [B, C, F, H, W]

        # Unscale
        if hasattr(self.vae.config, "scaling_factor"):
            latents = latents / self.vae.config.scaling_factor

        latents = latents.to(dtype=torch.float32)
        video = self.vae.decode(latents).sample

        # video: [B, C, F, H, W] in range [-1, 1]
        print(f"Output shape: {video.shape}")
        print(f"Output range: [{video.min().item():.3f}, {video.max().item():.3f}]")

        return DiffusionOutput(output=video)


# Simple test
if __name__ == "__main__":
    import urllib.request

    print("Testing AniSora I2V CogVideoX Pipeline...")

    # Create pipeline
    pipeline = AniSoraI2VCogVideoXPipeline(
        model_path="Disty0/Index-anisora-5B-diffusers",
        dtype=torch.bfloat16,
    )
    pipeline.to("cuda")

    # Download test image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    urllib.request.urlretrieve(url, "/tmp/cat.png")
    image = PIL.Image.open("/tmp/cat.png").convert("RGB")

    # Generate
    output = pipeline(
        prompt="a cat walking in the garden, high quality",
        image=image,
        negative_prompt="low quality, blurry",
        num_inference_steps=10,
        height=480,
        width=832,
        num_frames=17,
    )

    print(f"Output type: {type(output)}")
    print(f"Output.output shape: {output.output.shape}")

    # Check for NaN
    if torch.isnan(output.output).any():
        print("WARNING: Output contains NaN!")
    else:
        print("Output looks valid (no NaN)")

    # Save video
    from diffusers.utils import export_to_video

    video = output.output[0].permute(1, 2, 3, 0).cpu().numpy()  # [C, F, H, W] -> [F, H, W, C]
    video = ((video + 1) / 2 * 255).clip(0, 255).astype("uint8")
    export_to_video(video, "/workspace/test_cogvideox.mp4", fps=16)
    print("Video saved to /workspace/test_cogvideox.mp4")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AniSora V2/V3 Image-to-Video Pipeline using Wan2.1 architecture.

This pipeline uses a hybrid loading approach:
- VAE, Text Encoder, Scheduler from official Wan2.1-I2V-14B-Diffusers
- Transformer weights from AniSora (community conversion or official)

Supports:
- IndexTeam/Index-anisora (V2, V3.1, V3.2, anymask)
- aardsoul-music/Wan2.1-Anisora-14B
- ikusa/anisorav2

All these use WanModel architecture with in_dim=36 (16 noise + 16 image + 4 mask).
"""

from __future__ import annotations

import os

import PIL.Image
import torch
from diffusers import AutoencoderKLWan
from diffusers.models import WanTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    UMT5EncoderModel,
)

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler


# Default paths for components
DEFAULT_WAN_BASE = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DEFAULT_ANISORA_TRANSFORMER = "aardsoul-music/Wan2.1-Anisora-14B"


class AniSoraV2I2VPipeline(nn.Module):
    """
    AniSora V2/V3 Image-to-Video Pipeline using Wan2.1 architecture.

    This pipeline uses a hybrid loading approach for diffusers compatibility:
    - VAE, T5, CLIP, Scheduler from official Wan2.1 diffusers repo
    - Transformer weights from AniSora community conversions

    Args:
        model_path: Path to AniSora transformer weights (e.g., aardsoul-music/Wan2.1-Anisora-14B)
        wan_base_path: Path to Wan2.1 base model for VAE/T5/scheduler
        dtype: Model dtype (default: bfloat16)
        device: Target device
    """

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_ANISORA_TRANSFORMER,
        wan_base_path: str = DEFAULT_WAN_BASE,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        flow_shift: float = 5.0,
    ):
        super().__init__()
        self.device = device or get_local_device()
        self.dtype = dtype

        # Determine if local files
        local_anisora = os.path.exists(model_path)
        local_wan = os.path.exists(wan_base_path)

        print(f"=== AniSora V2 I2V Pipeline (Hybrid Loading) ===")
        print(f"AniSora transformer: {model_path}")
        print(f"Wan2.1 base (VAE/T5): {wan_base_path}")

        # Load tokenizer from Wan base
        print("Loading tokenizer from Wan2.1...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            wan_base_path,
            subfolder="tokenizer",
            local_files_only=local_wan,
        )

        # Load T5 text encoder from Wan base
        print("Loading T5 text encoder from Wan2.1...")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            wan_base_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_wan,
        )

        # Load CLIP image encoder from Wan base (for I2V conditioning)
        print("Loading CLIP image encoder from Wan2.1...")
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                wan_base_path,
                subfolder="image_processor",
                local_files_only=local_wan,
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                wan_base_path,
                subfolder="image_encoder",
                torch_dtype=dtype,
                local_files_only=local_wan,
            )
            self.has_image_encoder = True
        except Exception as e:
            print(f"Note: CLIP image encoder not available: {e}")
            self.image_processor = None
            self.image_encoder = None
            self.has_image_encoder = False

        # Load VAE from Wan base
        print("Loading VAE from Wan2.1...")
        self.vae = AutoencoderKLWan.from_pretrained(
            wan_base_path,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE in float32 for precision
            local_files_only=local_wan,
        )

        # Load transformer from AniSora weights
        # Note: aardsoul-music/Wan2.1-Anisora-14B uses old "WanModel" class
        # We need to load it with the correct config
        print(f"Loading transformer from AniSora: {model_path}...")
        
        # First, try loading with subfolder if it's structured like diffusers
        try:
            self.transformer = WanTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
                local_files_only=local_anisora,
            )
        except (OSError, ValueError):
            # If that fails, load directly but with correct config from Wan I2V base
            print("Using Wan2.1 I2V base config for transformer...")
            from diffusers import WanTransformer3DModel
            
            # Load config from Wan I2V base (which has in_channels=36)
            base_config = WanTransformer3DModel.load_config(
                wan_base_path,
                subfolder="transformer",
                local_files_only=local_wan,
            )
            
            # Create transformer with correct config
            self.transformer = WanTransformer3DModel.from_config(base_config)
            
            # Load weights from AniSora
            from safetensors.torch import load_file
            import glob
            import os as os_module
            
            # Find safetensor files
            if local_anisora:
                weight_path = model_path
            else:
                from huggingface_hub import snapshot_download
                weight_path = snapshot_download(model_path, local_files_only=False)
            
            safetensor_files = glob.glob(os_module.path.join(weight_path, "*.safetensors"))
            if not safetensor_files:
                safetensor_files = glob.glob(os_module.path.join(weight_path, "**/*.safetensors"), recursive=True)
            
            state_dict = {}
            for sf_path in safetensor_files:
                state_dict.update(load_file(sf_path))
            
            # Load state dict
            missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        
        self.transformer = self.transformer.to(dtype)

        # Initialize scheduler
        print("Initializing scheduler...")
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        # VAE scale factors
        self.vae_scale_factor_temporal = getattr(
            self.vae.config, "temporal_compression_ratio", 4
        )
        self.vae_scale_factor_spatial = getattr(
            self.vae.config, "spatial_compression_ratio", 8
        )

        self._current_timestep = None
        print("Pipeline loaded successfully!")

    def to(self, device):
        """Move pipeline to device."""
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        if self.has_image_encoder:
            self.image_encoder = self.image_encoder.to(device)
        return self

    @staticmethod
    def _prompt_clean(text: str) -> str:
        """Clean prompt text."""
        return " ".join(text.strip().split())

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 512,
    ):
        """Encode text prompts using T5."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        prompt_clean = [self._prompt_clean(p) for p in prompt]

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        # Trim and pad to consistent length
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # Negative prompt
        negative_prompt_embeds = None
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg = neg_text_inputs.input_ids.to(self.device)
            mask_neg = neg_text_inputs.attention_mask.to(self.device)
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()

            negative_prompt_embeds = self.text_encoder(
                ids_neg, mask_neg
            ).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.dtype, device=self.device
            )
            negative_prompt_embeds = [
                u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)
            ]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat(
                        [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]
                    )
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def encode_image_clip(self, image: PIL.Image.Image) -> torch.Tensor:
        """Encode image using CLIP for conditioning."""
        if not self.has_image_encoder:
            return None

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        image_embeds = self.image_encoder(pixel_values, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        last_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare latents for I2V generation.

        Returns:
            latents: Initial noise latents [B, C, F, H, W]
            condition: Encoded image condition with mask [B, C+4, F, H, W]
            first_frame_mask: Mask for conditioning
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )

        # Generate noise
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # Prepare image condition
        image = image.unsqueeze(2)  # [B, C, 1, H, W]

        if last_image is None:
            # Pad with zeros for remaining frames
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0], image.shape[1], num_frames - 1, height, width
                    ),
                ],
                dim=2,
            )
        else:
            # First and last frame conditioning
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0], image.shape[1], num_frames - 2, height, width
                    ),
                    last_image,
                ],
                dim=2,
            )

        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        # Encode through VAE
        latent_condition = self.vae.encode(video_condition).latent_dist.mode()
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        # Normalize latents
        if (
            hasattr(self.vae.config, "latents_mean")
            and self.vae.config.latents_mean is not None
        ):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, -1, 1, 1, 1
            ).to(latent_condition.device, latent_condition.dtype)
            latent_condition = (latent_condition - latents_mean) * latents_std

        latent_condition = latent_condition.to(dtype)

        # Create mask: 1 for frames with condition, 0 for frames to generate
        mask_lat_size = torch.ones(
            batch_size, 1, num_frames, latent_height, latent_width, device=device
        )
        if last_image is None:
            mask_lat_size[:, :, 1:] = 0  # Only first frame is conditioned
        else:
            mask_lat_size[:, :, 1:-1] = 0  # First and last frames are conditioned

        # Compress mask temporally
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = first_frame_mask.repeat(
            1, 1, self.vae_scale_factor_temporal, 1, 1
        )
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        # Concatenate mask with condition [B, C+4, F, H, W]
        condition = torch.cat([mask_lat_size, latent_condition], dim=1)

        # Return placeholder for first_frame_mask (not used in this mode)
        first_frame_mask = torch.ones(
            1,
            1,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
            device=device,
        )

        return latents, condition, first_frame_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image,
        negative_prompt: str | list[str] | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
        last_image: PIL.Image.Image | None = None,
        output_type: str = "tensor",
    ) -> DiffusionOutput:
        """
        Generate video from image and text prompt.

        Args:
            prompt: Text prompt(s)
            image: Input image (first frame)
            negative_prompt: Negative prompt(s)
            height: Output height
            width: Output width
            num_frames: Number of output frames (should be 4n+1)
            num_inference_steps: Denoising steps (40 recommended for I2V)
            guidance_scale: Classifier-free guidance scale
            generator: Random generator for reproducibility
            last_image: Optional last frame for interpolation
            output_type: "tensor", "np", or "pil"
        """
        # Ensure num_frames is compatible with VAE temporal scaling
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = (
                num_frames
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        # Encode prompt
        print("Encoding prompts...")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt
        )
        batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = (
            guidance_scale > 1.0 and negative_prompt_embeds is not None
        )

        # Encode image with CLIP for additional conditioning
        image_embeds = None
        if self.has_image_encoder:
            print("Encoding image with CLIP...")
            image_embeds = self.encode_image_clip(image)
            if image_embeds is not None:
                image_embeds = image_embeds.repeat(batch_size, 1, 1).to(self.dtype)

        # Preprocess image for VAE
        print("Preprocessing image for VAE...")
        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        image_tensor = video_processor.preprocess(image, height=height, width=width)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)

        # Handle last_image if provided
        last_image_tensor = None
        if last_image is not None:
            last_image_tensor = video_processor.preprocess(
                last_image, height=height, width=width
            )
            last_image_tensor = last_image_tensor.to(
                device=self.device, dtype=torch.float32
            )

        # Prepare latents
        print("Preparing latents...")
        num_channels_latents = self.transformer.config.out_channels  # 16

        latents, condition, first_frame_mask = self.prepare_latents(
            image=image_tensor,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            last_image=last_image_tensor,
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        print(f"Starting denoising loop ({num_inference_steps} steps)...")
        for i, t in enumerate(timesteps):
            self._current_timestep = t

            # Concatenate noise latents with condition [B, C, F, H, W] + [B, C+4, F, H, W]
            latent_model_input = torch.cat([latents, condition], dim=1).to(self.dtype)

            # Expand timestep
            timestep = t.expand(latents.shape[0])

            # Forward pass
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                return_dict=False,
            )[0]

            # Classifier-free guidance
            if do_classifier_free_guidance:
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % 10 == 0:
                print(f"  Step {i + 1}/{num_inference_steps}")

        self._current_timestep = None
        print("Decoding latents...")

        # Decode latents
        latents = latents.to(self.vae.dtype)

        # Denormalize
        if (
            hasattr(self.vae.config, "latents_mean")
            and self.vae.config.latents_mean is not None
        ):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, -1, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

        video = self.vae.decode(latents, return_dict=False)[0]

        print(f"Output shape: {video.shape}")
        print(f"Output range: [{video.min().item():.3f}, {video.max().item():.3f}]")

        return DiffusionOutput(output=video)


# Test
if __name__ == "__main__":
    import urllib.request

    print("Testing AniSora V2 I2V Pipeline (Hybrid Loading)...")

    # Create pipeline
    pipeline = AniSoraV2I2VPipeline(
        model_path="aardsoul-music/Wan2.1-Anisora-14B",
        wan_base_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        dtype=torch.bfloat16,
    )
    pipeline.to("cuda")

    # Download test image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    urllib.request.urlretrieve(url, "/tmp/cat.png")
    image = PIL.Image.open("/tmp/cat.png").convert("RGB")
    print(f"Input image size: {image.size}")

    # Generate
    output = pipeline(
        prompt="a cat walking in the garden, anime style, high quality",
        image=image,
        negative_prompt="low quality, blurry, text",
        num_inference_steps=20,  # Quick test
        height=480,
        width=832,
        num_frames=17,  # Short test
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

    video = (
        output.output[0].permute(1, 2, 3, 0).cpu().numpy()
    )  # [C, F, H, W] -> [F, H, W, C]
    video = ((video + 1) / 2 * 255).clip(0, 255).astype("uint8")
    export_to_video(video, "/workspace/test_anisora_v2.mp4", fps=16)
    print("Video saved to /workspace/test_anisora_v2.mp4")

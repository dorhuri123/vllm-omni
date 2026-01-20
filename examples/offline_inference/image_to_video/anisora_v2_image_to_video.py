# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: AniSora V2/V3 (14B) Image-to-Video Generation

This example demonstrates video generation using AniSora V2/V3 (14B model).
Uses hybrid loading approach:
- VAE/T5/CLIP from Wan2.1-I2V-14B-Diffusers
- Transformer from AniSora community conversion

Supported models:
- aardsoul-music/Wan2.1-Anisora-14B (recommended, ~65GB)
- ikusa/anisorav2

Requirements:
- ~65GB VRAM for 14B model in bfloat16
- ~35GB VRAM with 8-bit quantization (future support)

Usage:
    python anisora_v2_image_to_video.py --image input.png --prompt "anime scene"
"""

import argparse
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from PIL import Image

from vllm_omni.diffusion.models.anisora import AniSoraV2I2VPipeline


def main():
    parser = argparse.ArgumentParser(description="AniSora V2/V3 Image-to-Video")
    parser.add_argument(
        "--model",
        type=str,
        default="aardsoul-music/Wan2.1-Anisora-14B",
        help="AniSora transformer model path",
    )
    parser.add_argument(
        "--wan-base",
        type=str,
        default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        help="Wan2.1 base model for VAE/T5",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="anime style, high quality animation, smooth motion",
        help="Text prompt",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, static, text, watermark",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_v2.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Output width (default: 832)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output height (default: 480)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Number of frames (should be 4n+1, default: 81 = 5 seconds)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=40,
        help="Number of inference steps (default: 40)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale (default: 5.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Output video FPS (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    args = parser.parse_args()

    # Load image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"  Image size: {image.size}")

    # Create pipeline
    print(f"\nLoading AniSora V2 pipeline...")
    print(f"  Transformer: {args.model}")
    print(f"  Wan2.1 base: {args.wan_base}")

    pipeline = AniSoraV2I2VPipeline(
        model_path=args.model,
        wan_base_path=args.wan_base,
        dtype=torch.bfloat16,
        device=torch.device(args.device),
    )
    pipeline.to(args.device)

    # Set seed
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # Generate video
    print(f"\nGenerating video...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Steps: {args.num_steps}")

    output = pipeline(
        prompt=args.prompt,
        image=image,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    # Post-process and save
    print(f"\nSaving video to: {args.output}")
    video = output.output[0]  # [C, F, H, W]

    # Convert to [F, H, W, C] format
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    video = ((video + 1.0) / 2.0 * 255).clip(0, 255).astype("uint8")

    export_to_video(video, args.output, fps=args.fps)

    print(f"Done! Video saved with {args.num_frames} frames at {args.fps} FPS")
    duration = args.num_frames / args.fps
    print(f"Duration: {duration:.1f} seconds")


if __name__ == "__main__":
    main()

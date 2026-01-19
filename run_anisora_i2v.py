#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AniSora Image-to-Video generation script.

Usage:
    python run_anisora_i2v.py --image input.jpg --prompt "a cat playing with yarn"
    
    python run_anisora_i2v.py \
      --image input.jpg \
      --prompt "a cat playing with yarn" \
      --height 480 \
      --width 480 \
      --num_frames 9 \
      --num_inference_steps 20 \
      --guidance_scale 4.0 \
      --output output.mp4
"""

import argparse
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.outputs import OmniRequestOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a video from an image with AniSora I2V.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_anisora_i2v.py --image input.jpg --prompt "a cat playing"
  
  # Full parameters
  python run_anisora_i2v.py \\
    --image input.jpg \\
    --prompt "a cat playing with yarn" \\
    --negative_prompt "blurry" \\
    --height 480 \\
    --width 480 \\
    --num_frames 9 \\
    --num_inference_steps 20 \\
    --guidance_scale 4.0 \\
    --output output.mp4
        """,
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--prompt", required=True, help="Text prompt describing the desired motion."
    )
    parser.add_argument("--negative_prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="CFG scale for guidance.",
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Video height (divisible by 16)."
    )
    parser.add_argument(
        "--width", type=int, default=480, help="Video width (divisible by 16)."
    )
    parser.add_argument("--num_frames", type=int, default=9, help="Number of frames.")
    parser.add_argument(
        "--num_inference_steps", type=int, default=20, help="Sampling steps."
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help="Scheduler flow_shift.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anisora_i2v.mp4",
        help="Path to save the video (mp4).",
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="Frames per second for output video."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Load input image (handle both URLs and local paths)
    print(f"\n{'='*70}")
    print(f"📥 PHASE 1: Loading image")
    print(f"{'='*70}")
    print(f"   Source: {args.image}")
    if args.image.startswith("http://") or args.image.startswith("https://"):
        import requests
        from io import BytesIO

        print("   Downloading image from URL...")
        response = requests.get(args.image, timeout=30)
        image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
        print(f"   ✅ Downloaded successfully (size: {len(response.content) / 1024 / 1024:.2f} MB)")
    else:
        print("   Loading local image...")
        image = PIL.Image.open(args.image).convert("RGB")
        print(f"   ✅ Loaded successfully")

    # Resize to target dimensions
    print(f"   Resizing to: {args.width}x{args.height}...")
    image = image.resize((args.width, args.height), PIL.Image.Resampling.LANCZOS)
    print(f"   ✅ Image ready\n")

    # Initialize Omni with AniSora I2V
    print(f"{'='*70}")
    print(f"🚀 PHASE 2: Initializing AniSora I2V Pipeline")
    print(f"{'='*70}")
    print(f"   Model: Disty0/Index-anisora-5B-diffusers")
    print(f"   Model size: 21.5 GB")
    print(f"   VAE slicing: True | VAE tiling: True")
    print(f"   Flow shift: {args.flow_shift}")
    print(f"   (First run may take 10-15 min to download model...)")
    print(f"   Waiting for pipeline initialization...")
    omni = Omni(
        model="Disty0/Index-anisora-5B-diffusers",
        model_class_name="AniSoraImageToVideoPipeline",
        vae_use_slicing=True,
        vae_use_tiling=True,
        flow_shift=args.flow_shift,
        stage_init_timeout=1200,  # 20 minutes for first-time model download + initialization
    )
    print(f"   ✅ Pipeline initialized successfully\n")

    # Generate video
    print(f"{'='*70}")
    print(f"🎬 PHASE 3: Generating Video")
    print(f"{'='*70}")
    print(f"   Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"   Negative prompt: {args.negative_prompt}")
    print(f"   Resolution: {args.width}x{args.height}")
    print(f"   Frames: {args.num_frames}")
    print(f"   Inference steps: {args.num_inference_steps}")
    print(f"   Guidance scale: {args.guidance_scale}")
    print(f"   Starting generation...")

    frames = omni.generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        pil_image=image,
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )
    print(f"   ✅ Generation complete\n")

    # Extract video frames from OmniRequestOutput
    print(f"{'='*70}")
    print(f"📦 PHASE 4: Processing Output")
    print(f"{'='*70}")
    print(f"   Extracting frames from output...")
    if isinstance(frames, list) and len(frames) > 0:
        first_item = frames[0]

        if hasattr(first_item, "final_output_type"):
            if first_item.final_output_type != "image":
                raise ValueError(
                    f"Unexpected output type '{first_item.final_output_type}', expected 'image' for video generation."
                )

            if (
                hasattr(first_item, "is_pipeline_output")
                and first_item.is_pipeline_output
            ):
                if (
                    isinstance(first_item.request_output, list)
                    and len(first_item.request_output) > 0
                ):
                    inner_output = first_item.request_output[0]
                    if isinstance(inner_output, OmniRequestOutput) and hasattr(
                        inner_output, "images"
                    ):
                        frames = inner_output.images[0] if inner_output.images else None
                        if frames is None:
                            raise ValueError("No video frames found in output.")
            elif hasattr(first_item, "images") and first_item.images:
                frames = first_item.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    print(f"   ✅ Frames extracted\n")

    # Save video
    print(f"{'='*70}")
    print(f"💾 PHASE 5: Saving Video")
    print(f"{'='*70}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"   Output path: {output_path}")

    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

    print(f"   Converting frames to video tensor...")
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    else:
        video_array = frames
        if hasattr(video_array, "shape") and video_array.ndim == 5:
            video_array = video_array[0]

    if isinstance(video_array, np.ndarray) and video_array.ndim == 4:
        video_array = list(video_array)

    print(f"   Exporting to MP4 (fps={args.fps})...")
    export_to_video(video_array, str(output_path), fps=args.fps)
    print(f"   ✅ Video saved\n")

    print(f"{'='*70}")
    print(f"🎉 SUCCESS - AniSora I2V generation complete!")
    print(f"{'='*70}")
    print(f"   Output: {output_path}")
    print(f"   FPS: {args.fps}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

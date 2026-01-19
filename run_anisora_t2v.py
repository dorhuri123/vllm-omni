#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AniSora Text-to-Video generation script.

Note: Available diffusers models (Disty0/Index-anisora-5B-diffusers) are I2V (Image-to-Video).
For T2V, use with --image flag to provide a reference image.

Usage:
    python run_anisora_t2v.py --prompt "a cat walking in the park"
    
    python run_anisora_t2v.py \
      --prompt "Two cats boxing" \
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
        description="Generate a video with AniSora (T2V or I2V).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Note: Current diffusers models are I2V, provide a reference image
  python run_anisora_t2v.py \\
    --image reference.jpg \\
    --prompt "a cat playing with yarn" \\
    --height 480 \\
    --width 480 \\
    --num_frames 9 \\
    --guidance_scale 4.0 \\
    --output video.mp4
        """,
    )
    parser.add_argument(
        "--prompt", required=True, help="Text prompt for video generation."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="(Optional) Reference image for I2V. If provided, generates I2V instead of T2V.",
    )
    parser.add_argument("--negative_prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="CFG scale.",
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
        default="anisora_output.mp4",
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

    # Load reference image if provided (for I2V)
    print(f"\n{'='*70}")
    print(f"📥 PHASE 1: Loading Input")
    print(f"{'='*70}")
    
    image = None
    if args.image:
        print(f"   Loading reference image: {args.image}")
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
        image = image.resize((args.width, args.height), PIL.Image.Resampling.LANCZOS)
        print(f"   Resizing to: {args.width}x{args.height}...")
        print(f"   ✅ Image ready")
        generation_type = "I2V (Image-to-Video)"
    else:
        generation_type = (
            "T2V (Text-to-Video) - Using I2V model with generated reference"
        )
        print("⚠️  No image provided, using I2V model in text-only mode")
        print(f"   Mode: {generation_type}")
    print()

    # Initialize Omni
    print(f"{'='*70}")
    print(f"🚀 PHASE 2: Initializing AniSora {generation_type}")
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
    if image is not None:
        print(f"   Mode: Image-to-Video (using provided reference)")
    else:
        print(f"   Mode: Text-only (reference will be generated)")
    print(f"   Starting generation...")

    generate_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "generator": generator,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_frames": args.num_frames,
    }

    # Add image if provided
    if image is not None:
        generate_kwargs["pil_image"] = image

    frames = omni.generate(**generate_kwargs)
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
                    f"Unexpected output type '{first_item.final_output_type}', expected 'image'."
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
    print(f"🎉 SUCCESS - AniSora generation complete!")
    print(f"{'='*70}")
    print(f"   Output: {output_path}")
    print(f"   FPS: {args.fps}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

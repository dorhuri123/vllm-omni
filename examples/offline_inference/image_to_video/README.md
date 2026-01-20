# Image-To-Video

This example demonstrates how to generate videos from images using vLLM-Omni's offline inference API.

## Supported Models

- **Wan2.2-I2V-A14B-Diffusers** (MoE) - Alibaba's Wan2.2 14B MoE model
- **Wan2.2-TI2V-5B-Diffusers** (Unified) - Alibaba's unified T2V+I2V 5B model
- **AniSora V1 (5B)** - CogVideoX-based anime video generation
- **AniSora V2/V3 (14B)** - Wan2.1-based anime video generation

---

## AniSora V1 (5B) - CogVideoX-based

Optimized for anime-style video generation using CogVideoX architecture.

```bash
python anisora_image_to_video.py \
  --model IndexTeam/AniSora-v1-i2v-diffusers \
  --image input.png \
  --prompt "anime girl walking, flowing hair, studio ghibli style" \
  --height 480 \
  --width 720 \
  --num_frames 49 \
  --guidance_scale 5.0 \
  --num_inference_steps 50 \
  --fps 16 \
  --output anisora_v1.mp4
```

**Requirements:** ~24GB VRAM

---

## AniSora V2/V3 (14B) - Wan2.1-based

High-quality anime video generation using Wan2.1 architecture with community weights.

```bash
python anisora_v2_image_to_video.py \
  --image input.png \
  --prompt "anime scene, high quality animation, smooth motion" \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --guidance-scale 5.0 \
  --num-inference-steps 30 \
  --fps 8 \
  --output anisora_v2.mp4
```

**Requirements:** ~65GB VRAM for 14B model in bfloat16

**Supported transformer models:**
- `aardsoul-music/Wan2.1-Anisora-14B` (recommended)
- `ikusa/anisorav2`

---

## Wan2.2 Models

### Wan2.2-I2V-A14B-Diffusers (MoE)
```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image input.png \
  --prompt "A cat playing with yarn, smooth motion" \
  --negative_prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num_frames 48 \
  --guidance_scale 5.0 \
  --guidance_scale_high 6.0 \
  --num_inference_steps 40 \
  --boundary_ratio 0.875 \
  --flow_shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)
```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image input.png \
  --prompt "A cat playing with yarn, smooth motion" \
  --negative_prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num_frames 48 \
  --guidance_scale 4.0 \
  --num_inference_steps 40 \
  --flow_shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V).
- `--image`: Path to input image (required).
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set). Dimensions should be multiples of 16.
- `--num_frames`: Number of frames (default 81).
- `--guidance_scale` and `--guidance_scale_high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative_prompt`: Optional list of artifacts to suppress.
- `--boundary_ratio`: Boundary split ratio for two-stage MoE models.
- `--flow_shift`: Scheduler flow shift (5.0 for 720p, 12.0 for 480p).
- `--num_inference_steps`: Number of denoising steps (default 50).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: Path to save the generated video.

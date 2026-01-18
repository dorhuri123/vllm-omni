# AniSora Integration - Complete File Documentation

## Overview
This document explains all files created for AniSora T2V and I2V model support in vLLM-Omni. AniSora is an anime-specialized video generation model that extends Wan2.2's architecture with improved animation quality.

---

## 📁 File Structure

```
vllm_omni/diffusion/models/anisora/
├── __init__.py                    (Module initialization)
├── pipeline_anisora.py            (T2V: Text-to-Video pipeline)
├── pipeline_anisora_i2v.py        (I2V: Image-to-Video pipeline)

examples/offline_inference/
├── text_to_video/
│   └── anisora_text_to_video.py   (T2V CLI example)
├── image_to_video/
│   └── anisora_image_to_video.py  (I2V CLI example)

tests/diffusion/models/
└── test_anisora_registry.py       (Registry validation tests)

docs/
└── models/supported_models.md     (Documentation update)
```

---

## 🔍 Detailed File Explanations

### 1. **vllm_omni/diffusion/models/anisora/__init__.py**

**Purpose:** Module initialization stub

**Content:** Empty file that marks `anisora` as a Python package

**Why it exists:** Python requires `__init__.py` to recognize directories as packages, enabling imports like:
```python
from vllm_omni.diffusion.models.anisora.pipeline_anisora import AniSoraPipeline
```

**How it works:** N/A (placeholder file)

---

### 2. **vllm_omni/diffusion/models/anisora/pipeline_anisora.py**

**Purpose:** Implements the AniSora Text-to-Video (T2V) pipeline

**Key Components:**

#### **2.1 Pre/Post-Process Functions**

```python
def get_anisora_post_process_func(od_config):
    """Converts model output latents → video frames
    
    Returns: VideoProcessor that decodes latent representations into playable video
    """

def get_anisora_pre_process_func(od_config):
    """Prepares input requests for T2V (no-op for T2V)
    
    For T2V, input is just text, so no preprocessing needed
    """
```

**Why they exist:**
- vLLM-Omni uses a registry pattern where each pipeline has pre/post-process functions
- Pre-process: Validates/prepares inputs before inference
- Post-process: Converts raw model outputs to user-friendly formats

---

#### **2.2 AniSoraPipeline Class**

**Purpose:** Main T2V inference engine

**Constructor (`__init__`):**
```python
AniSoraPipeline(od_config, prefix="")
```

**What it loads:**
1. **Tokenizer** (AutoTokenizer) - Converts text prompts → token IDs
2. **Text Encoder** (UMT5EncoderModel) - Converts tokens → semantic embeddings (768-dim)
3. **VAE** (AutoencoderKLWan) - Compresses video frames into latent space (8x spatial, 4x temporal)
4. **Transformer** (WanTransformer3DModel) - Denoising network (Diffusion Transformer)
5. **Scheduler** (FlowUniPCMultistepScheduler) - Controls noise schedule during inference

**Why each component:**
- **Tokenizer+Encoder**: Understand what the user wants to generate
- **VAE**: Work in compressed latent space (faster, less VRAM)
- **Transformer**: The "denoiser" that gradually removes noise from random video
- **Scheduler**: Tells transformer when to apply how much denoising

---

#### **2.3 Key Methods**

**`forward(req, prompt, negative_prompt, ...)`**
- **Purpose**: Generate a video from text
- **Inputs**:
  - `prompt` (required): What to generate (e.g., "A cat playing with yarn")
  - `negative_prompt` (optional): What to avoid
  - `height`, `width`: Output resolution (auto-adjusted to VAE alignment)
  - `num_frames`: Video length (default 81 frames ≈ 3 seconds)
  - `num_inference_steps`: Quality vs speed tradeoff (more = slower + better)
  - `guidance_scale`: How strongly to follow prompt (default 4.0, range 1.0-15.0)

- **Process**:
  1. Encode text prompt → semantic embedding
  2. Create random noise latents (same shape as compressed video)
  3. Loop through timesteps (scheduler):
     - Predict noise in current latent
     - Apply guidance: blend conditional (prompted) + unconditional (unguided) predictions
     - Update latents by removing predicted noise
  4. Decode final latents → video frames
  5. Return video

- **Output**: Video tensor (B, C, F, H, W) where B=batch, C=channels(3), F=frames, H/W=resolution

---

**`_encode_prompt(prompt, negative_prompt, ...)`**
- Tokenizes text → embeddings using UMT5
- Handles batch processing and sequence length padding
- Supports classifier-free guidance (CFG):
  - Creates both prompted + unprompted embeddings
  - During inference, blends: `output = unguided + guidance_scale * (guided - unguided)`

---

**`_prepare_latents(batch_size, num_channels, height, width, num_frames, ...)`**
- Creates random noise tensor for diffusion
- Respects VAE scale factors:
  - Spatial: divides H/W by 8
  - Temporal: divides frames by 4
  - Example: 720p, 81 frames → latent shape (16, 90, 20, 10, 160, 240)

---

**`_check_inputs(...)`**
- Validates user inputs
- Ensures height/width divisible by 16 (VAE requirement)
- Prevents conflicting arguments

---

**`_load_transformer_config(model_path, subfolder, ...)`**
- Loads transformer config JSON from local disk or HuggingFace
- Returns dict with architecture details

---

**`_create_transformer_from_config(config)`**
- Builds WanTransformer3DModel from config
- Maps config keys to transformer constructor args

---

**`load_weights(weights)`**
- Loads pre-trained model weights using vLLM's AutoWeightsLoader
- Called during inference engine startup

---

### 3. **vllm_omni/diffusion/models/anisora/pipeline_anisora_i2v.py**

**Purpose:** Implements AniSora Image-to-Video (I2V) pipeline

**Key Differences from T2V:**

1. **Input**: Image + optional text prompt
2. **Process**: Use first frame as "anchor" that doesn't change much

---

#### **3.1 Pre/Post-Process Functions**

```python
def get_anisora_i2v_pre_process_func(od_config):
    """Loads image from disk if needed
    
    Converts image_path → PIL.Image
    """

def get_anisora_i2v_post_process_func(od_config):
    """Same as T2V: decodes latents → video"""
```

---

#### **3.2 AniSoraI2VPipeline Class**

**Constructor:** Similar to T2V, loads all same components

**Key Method: `forward(...)`**

**New inputs:**
- `pil_image`: Input image (PIL.Image)
- `image_path`: Path to input image (alternative to pil_image)
- `prompt`: Optional text to guide motion

**New process steps:**

1. **Encode prompt** (same as T2V)

2. **Encode input image to latent space**:
   ```
   image (H×W×3) → VAE encoder → latent_condition (1×channels×1×H/8×W/8)
   ```

3. **Normalize latent to model space**:
   - Apply learnable mean/std from VAE config
   - Ensures latent is in DiT's expected range

4. **Create first-frame mask**:
   ```python
   mask = torch.ones(1, 1, num_frames, ...)  # All 1s initially
   mask[:, :, 0] = 0  # Frame 0 is 0
   ```

5. **Denoising loop with conditioning blend**:
   ```python
   for timestep in timesteps:
       # Blend: keep frame 0 as image, let others denoise
       latent_input = (1 - mask) * latent_condition + mask * latents
       # Predict noise with optional text guidance
       noise_pred = transformer(latent_input, timestep, text_embedding)
       # Update latents (remove predicted noise)
       latents = scheduler.step(noise_pred, timestep, latents)
   ```

6. **Final blending**: Ensure frame 0 stays close to input image
7. **Decode**: latents → video frames

**Why the masking strategy?**
- Without it: Generated video could completely ignore input image
- With masking: Frame 0 stays anchored, other frames interpolate naturally from it
- Creates smooth motion continuation from static image

---

### 4. **examples/offline_inference/text_to_video/anisora_text_to_video.py**

**Purpose:** Runnable CLI script for T2V inference

**What it does:**
1. Parses command-line arguments
2. Creates Omni inference engine
3. Calls `omni.generate()` with parameters
4. Extracts video frames from response
5. Exports to MP4 using diffusers' `export_to_video()`

**CLI Arguments:**
```bash
python anisora_text_to_video.py \
  --model /path/to/model \
  --prompt "A serene lakeside sunset" \
  --negative_prompt "ugly, blurry" \
  --seed 42 \
  --guidance_scale 4.0 \
  --height 720 \
  --width 1280 \
  --num_frames 81 \
  --num_inference_steps 40 \
  --output output.mp4 \
  --fps 24
```

**Why it exists:** Demonstrates how to use AniSora in production (offline inference mode)

---

### 5. **examples/offline_inference/image_to_video/anisora_image_to_video.py**

**Purpose:** Runnable CLI script for I2V inference

**Key differences from T2V:**
- Requires `--image` argument (input image path)
- Auto-calculates resolution from image if not specified
- Uses `pil_image=image` in `omni.generate()` call

**CLI Arguments:**
```bash
python anisora_image_to_video.py \
  --model /path/to/model \
  --image input.jpg \
  --prompt "Camera zoom out" \
  --seed 42 \
  --guidance_scale 5.0 \
  --num_inference_steps 50 \
  --output output.mp4
```

**Helper Function: `calculate_dimensions(image, max_area)`**
- Auto-determines output resolution maintaining aspect ratio
- Default max area: 480 × 832 pixels (for I2V typically lower than T2V)

---

### 6. **tests/diffusion/models/test_anisora_registry.py**

**Purpose:** Validates that AniSora pipelines are correctly registered

**Test:**
```python
def test_anisora_registry_entries_present():
    assert "AniSoraPipeline" in PIPELINE_REGISTRY
    assert "AniSoraImageToVideoPipeline" in PIPELINE_REGISTRY
    # ... check pre/post-process functions mapped
```

**Why it exists:**
- Registry is the lookup table for available pipelines
- This test ensures AniSora entries are registered before deployment
- Catches configuration errors early

**How it works:**
- Imports vLLM-Omni registry
- Checks that AniSora entries exist
- Verifies pre/post-process functions are mapped

---

### 7. **docs/models/supported_models.md** (Updated)

**Purpose:** Documentation of supported models

**Change:** Added two rows to the GPU/AMD table:
```markdown
|`AniSoraPipeline` | AniSora-T2V | Index-Anisora/AniSora-v3.1-T2V |
|`AniSoraImageToVideoPipeline` | AniSora-I2V | Index-Anisora/AniSora-v3.1-I2V |
```

**Why it exists:** Users need to know what models vLLM-Omni supports

---

## 🔗 Integration Points

### How Registry Works
1. **vllm_omni/diffusion/registry.py** contains mappings:
   ```python
   PIPELINE_REGISTRY = {
       "AniSoraPipeline": ("anisora", "pipeline_anisora", "AniSoraPipeline"),
       "AniSoraImageToVideoPipeline": ("anisora", "pipeline_anisora_i2v", "AniSoraI2VPipeline"),
   }
   ```

2. When user requests model, vLLM-Omni:
   - Looks up pipeline name in registry
   - Dynamically imports module: `vllm_omni.diffusion.models.anisora.pipeline_anisora`
   - Instantiates class: `AniSoraPipeline(od_config)`

3. Pre/post-process functions are mapped:
   ```python
   DIFFUSION_PRE_PROCESS_MAP = {
       "AniSoraPipeline": "get_anisora_pre_process_func",
       ...
   }
   ```

---

## 🛠️ Data Flow

### T2V Generation
```
User Input (text)
    ↓
get_anisora_pre_process_func (no-op)
    ↓
OmniDiffusionRequest created
    ↓
AniSoraPipeline.forward()
    ├─ Encode prompt → embeddings
    ├─ Prepare random latents
    ├─ Loop through denoising steps
    │   ├─ Predict noise with guidance
    │   └─ Update latents
    └─ Decode latents → video
    ↓
get_anisora_post_process_func (decode to frames)
    ↓
Export to MP4
```

### I2V Generation
```
User Input (image + optional text)
    ↓
get_anisora_i2v_pre_process_func (load image)
    ↓
OmniDiffusionRequest created
    ↓
AniSoraI2VPipeline.forward()
    ├─ Encode prompt + image to latent
    ├─ Create first-frame mask
    ├─ Prepare noise latents
    ├─ Loop through denoising steps
    │   ├─ Blend: keep frame 0, denoise others
    │   ├─ Predict noise with guidance
    │   └─ Update latents
    └─ Final blend + decode
    ↓
get_anisora_i2v_post_process_func
    ↓
Export to MP4
```

---

## 🐛 Errors Fixed

| Error | Cause | Fix |
|-------|-------|-----|
| Unsorted imports | Linter requirement | Alphabetized: `diffusers` → `torch` → `transformers` |
| Missing `Iterable` | Type annotation needed | Added `from collections.abc import Iterable` |
| Line too long (120 char limit) | Code readability | Broke long error messages into multiple lines |
| Long function signature | PEP 8 style | Multi-line signature with proper indentation |

---

## 📊 Comparison: T2V vs I2V

| Aspect | T2V | I2V |
|--------|-----|-----|
| **Input** | Text only | Image + optional text |
| **Pre-process** | No-op | Load image from disk |
| **Key difference** | Free generation | Conditioned on image frame |
| **Masking** | None | First-frame anchor mask |
| **Use case** | Create video from scratch | Animate existing image |
| **Guidance scale** | Typically 4.0 | Typically 5.0 |

---

## 🚀 Usage Example

**Text-to-Video:**
```bash
cd vllm-omni
python examples/offline_inference/text_to_video/anisora_text_to_video.py \
  --model /path/to/anisora-v3.1 \
  --prompt "Anime girl walking through a cherry blossom garden" \
  --output anime_walk.mp4
```

**Image-to-Video:**
```bash
python examples/offline_inference/image_to_video/anisora_image_to_video.py \
  --model /path/to/anisora-v3.1-i2v \
  --image static_anime.jpg \
  --prompt "Camera pans left" \
  --output anime_pan.mp4
```

---

## ✅ Summary

| File | Type | Purpose | Key Classes/Functions |
|------|------|---------|----------------------|
| `__init__.py` | Init | Package marker | (empty) |
| `pipeline_anisora.py` | Pipeline | T2V inference | `AniSoraPipeline`, `get_anisora_*_process_func` |
| `pipeline_anisora_i2v.py` | Pipeline | I2V inference | `AniSoraI2VPipeline`, `get_anisora_i2v_*_process_func` |
| `anisora_text_to_video.py` | CLI | T2V example | `parse_args()`, `main()` |
| `anisora_image_to_video.py` | CLI | I2V example | `parse_args()`, `main()` |
| `test_anisora_registry.py` | Test | Validation | `test_anisora_registry_entries_present()` |
| `supported_models.md` | Doc | Model list | (markdown table) |

All files work together to provide seamless AniSora video generation in vLLM-Omni!

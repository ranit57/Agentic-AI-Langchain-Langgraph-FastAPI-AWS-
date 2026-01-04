# Text-to-Image Generation using Stable Diffusion and Diffusers

## Overview
This project demonstrates how to generate images from text prompts using **Stable Diffusion** and the **Hugging Face Diffusers** library. It leverages pre-trained diffusion models to create high-quality, photorealistic images based on detailed text descriptions.

## Features
- **Text-to-Image Generation**: Convert natural language prompts into stunning visual images
- **Multiple Model Support**: Uses both dreamlike-art and Stable Diffusion XL models
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs for faster image generation
- **Customizable Parameters**: Control image generation with:
  - Negative prompting
  - Inference steps
  - Image dimensions (height/width)
  - Number of images per prompt
  - Guidance scale

## Installation
```bash
pip install diffusers transformers accelerate torch
```

## Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch with CUDA support
- Hugging Face Transformers library
- Matplotlib (for visualization)

## Models Used
1. **dreamlike-art/dreamlike-diffusion-1.0** - Creative, artistic style
2. **stabilityai/stable-diffusion-xl-base-1.0** - High-quality realistic images

## Key Components

### 1. Pipeline Setup
```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)
pipe = pipe.to("cuda")
```

### 2. Image Generation Function
The `generate_image()` function handles:
- Accepting prompts and generation parameters
- Generating single or multiple images
- Visualizing results in a subplot layout

### 3. Example Usage
```python
prompt = "dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions..."
image = pipe(prompt).images[0]
```

## Workflow
1. Install required dependencies
2. Load a pre-trained diffusion model from Hugging Face Hub
3. Prepare text prompts describing desired images
4. Call the pipeline with prompts and parameters
5. Visualize generated images using Matplotlib

## Output
- Generated PIL images can be:
  - Displayed inline in notebooks
  - Saved to disk
  - Used in downstream applications

## Performance Tips
- Use `float16` precision for faster generation and lower memory usage
- Increase `num_inference_steps` (typically 20-50) for better quality but slower generation
- GPU with VRAM 8GB+ recommended for optimal performance

## Limitations
- Quality depends on prompt clarity and specificity
- Generation time varies by model and hardware
- Output may contain artifacts or unwanted details

## References
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/using-diffusers/loading)
- [Stable Diffusion Model](https://huggingface.co/spaces/stabilityai/stable-diffusion)
- [Dreamlike Art Model](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0)

## Author
**Ranit Pal** - Data Science & AI Learning Project
- GitHub: [@ranit57](https://github.com/ranit57)
- Email: ranitpal57@gmail.com

---
**Last Updated**: January 2026

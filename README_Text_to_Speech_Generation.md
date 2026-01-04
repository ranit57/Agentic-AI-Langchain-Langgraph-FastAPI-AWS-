# Text-to-Speech Generation with LLMs and Hugging Face

## Overview
This notebook demonstrates how to generate speech from text using Hugging Face models and pipelines. It shows how to use text-to-speech models (e.g., `suno/bark-small`) to synthesize audio from plain text and play it in a notebook environment.

## Features
- Convert text strings to audio waveform
- Use Hugging Face `pipeline("text-to-speech")`
- Support for GPU acceleration using CUDA
- Playback using IPython `Audio` in notebooks

## Installation
```bash
pip install transformers
```

## Requirements
- Python 3.8+
- Transformers
- PyTorch (with CUDA for GPU)
- IPython for audio playback in notebooks

## Example
```python
from transformers import pipeline
from IPython.display import Audio

pipe = pipeline("text-to-speech", model="suno/bark-small", device="cuda")
text = "Python is a high-level, general-purpose programming language."
output = pipe(text)
Audio(output['audio'], rate=output['sampling_rate'])
```

## Notes
- Model weights may be large; expect downloads on first run.
- Ensure GPU device index is available and CUDA is configured.

## Author
**Ranit Pal** - Data Science & AI Learning Project
- GitHub: [@ranit57](https://github.com/ranit57)
- Email: ranitpal57@gmail.com

---
**Last Updated**: January 2026

# HuggingFace Transformers Demo

## Overview
This comprehensive demo showcases the **Hugging Face Transformers** library and its powerful `pipeline` API. It demonstrates how to use pre-trained models for various Natural Language Processing (NLP) and Computer Vision tasks without extensive coding.

## Features
- **NLP Task Pipelines**: 9+ NLP demonstrations
- **Computer Vision Pipelines**: Image classification and object detection
- **Easy-to-Use API**: One-line model loading and inference
- **Pre-trained Models**: Ready-to-use models from Hugging Face Hub

## Installation
```bash
pip install transformers
```

## Requirements
- Python 3.8+
- Transformers library
- PyTorch or TensorFlow
- NVIDIA CUDA support (for GPU acceleration)

## NLP Tasks Covered

### 1. Text Classification
Assigns categories to text (sentiment analysis, spam detection, topic classification)
```python
classifier = pipeline("text-classification")
```

### 2. Token Classification
Assigns labels to individual tokens (Named Entity Recognition, POS tagging)
```python
token_classifier = pipeline("token-classification")
```

### 3. Question Answering
Extracts answers from given context based on questions
```python
question_answerer = pipeline("question-answering")
```

### 4. Text Generation
Generates text based on prompts (language modeling, story generation)
```python
text_generator = pipeline("text-generation")
```

### 5. Summarization
Condenses long documents into shorter summaries
```python
summarizer = pipeline("summarization")
```

### 6. Translation
Translates text between languages
```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
```

### 7. Text-to-Text Generation
General-purpose text transformation (summarization, translation, paraphrase)
```python
text2text_generator = pipeline("text2text-generation")
```

### 8. Fill-Mask
Predicts masked tokens in sequences (like BERT's masked language modeling)
```python
fill_mask = pipeline("fill-mask")
```

### 9. Feature Extraction
Extracts hidden states or embeddings from text
```python
feature_extractor = pipeline("feature-extraction")
```

### 10. Sentence Similarity
Measures similarity between two sentences
```python
sentence_similarity = pipeline("sentence-similarity")
```

## Computer Vision Tasks Covered

### 1. Image Classification
Classifies the main content of images
```python
image_classifier = pipeline("image-classification")
```

### 2. Object Detection
Identifies objects and bounding boxes in images
```python
object_detector = pipeline("object-detection")
```

## Quick Start

### Basic Usage Pattern
```python
from transformers import pipeline

# Create pipeline
task_pipeline = pipeline("task_name")

# Use pipeline
result = task_pipeline("your_input_here")

# Get results
print(result)
```

### Example: Sentiment Analysis
```python
classifier = pipeline("text-classification")
result = classifier("This movie is absolutely fantastic!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

## How Pipelines Work
1. **Load Model**: Downloads pre-trained weights from Hugging Face Hub
2. **Tokenize Input**: Converts text to token IDs
3. **Run Inference**: Passes tokens through the model
4. **Post-process**: Converts model outputs to human-readable format

## Advantages
- ✅ No need to handle tokenization manually
- ✅ Automatic GPU/CPU detection
- ✅ Easy to switch between models
- ✅ Extensive model zoo available
- ✅ Production-ready implementations

## Performance Considerations
- First run downloads model weights (can be large)
- GPU acceleration recommended for batch processing
- CPU-based inference is slower but works fine for demos

## Available Models
Visit [Hugging Face Model Hub](https://huggingface.co/models) to explore:
- BERT, RoBERTa, ALBERT for NLP tasks
- ViT, ResNet for vision tasks
- Multilingual and domain-specific models

## Use Cases
- **Sentiment Analysis**: Customer feedback analysis
- **Named Entity Recognition**: Information extraction
- **Machine Translation**: Real-time translation services
- **Text Summarization**: Document condensing
- **Image Classification**: Visual content categorization
- **Object Detection**: Visual search and analysis

## References
- [Hugging Face Official Documentation](https://huggingface.co/docs/transformers)
- [Transformers GitHub Repository](https://github.com/huggingface/transformers)
- [Model Hub](https://huggingface.co/models)

## Author
Data Science & AI Learning Project

---
**Last Updated**: January 2026

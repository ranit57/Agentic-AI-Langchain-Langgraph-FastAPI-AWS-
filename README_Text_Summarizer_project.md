# Text Summarizer Project using Transformers

## Overview
This project demonstrates **Abstractive Text Summarization** using Google's **PEGASUS** model and the **SAMSum** dataset. It automatically condenses conversations and long documents into concise, meaningful summaries using state-of-the-art transformer-based neural networks.

## Features
- **Abstractive Summarization**: Generates human-like summaries (not just extraction)
- **Pre-trained PEGASUS Model**: Fine-tuned on conversation summarization
- **SAMSum Dataset**: Conversation and summary pairs for training/evaluation
- **GPU Acceleration**: CUDA support for faster inference
- **Evaluation Metrics**: ROUGE score calculation for quality assessment
- **Flexible Input**: Works with various text lengths and formats

## Installation
```bash
pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr accelerate -q
```

## Requirements
- Python 3.8+
- Transformers library (Hugging Face)
- Datasets library (for SAMSum dataset)
- PyTorch with CUDA support (optional but recommended)
- NLTK for tokenization
- Rouge-score for evaluation

## Dataset

### SAMSum Dataset
- **Format**: Conversational dialogues with summaries
- **Size**: 16,369 samples (train: 14,732 | validation: 818 | test: 819)
- **Features**: 'dialogue' (conversation) and 'summary' (target summary)
- **Use Case**: Fine-tuning summarization models

```python
from datasets import load_dataset
dataset_samsum = load_dataset("samsum")

# Example:
# Dialogue: "Mike: Hey, how are you?..."
# Summary: "Mike and Jennifer discussed their weekend plans..."
```

## Workflow

### 1. Environment Setup
```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# GPU availability check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

### 2. Model Loading
```python
# Google PEGASUS model for dialogue summarization
model_ckpt = "google/pegasus-cnn_dailymail"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
```

### 3. Data Exploration
```python
dataset_samsum = load_dataset("samsum")

# Get split information
split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum]
# Output: [14732, 818, 819] for train/validation/test

# View dataset features
print(dataset_samsum['train'].column_names)
# Output: ['id', 'dialogue', 'summary']

# Examine sample
print(dataset_samsum["test"][1]["dialogue"])
print(dataset_samsum["test"][1]["summary"])
```

### 4. Feature Engineering
```python
def convert_examples_to_features(example_batch):
    # Tokenize input dialogues
    input_encodings = tokenizer(
        example_batch['dialogue'],
        max_length=1024,
        truncation=True
    )
    
    # Tokenize target summaries
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch['summary'],
            max_length=128,
            truncation=True
        )
    
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

# Apply to dataset
dataset_samsum_processed = dataset_samsum.map(
    convert_examples_to_features,
    batched=True
)
```

### 5. Model Training
```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_samsum_processed['train'],
    eval_dataset=dataset_samsum_processed['validation'],
    data_collator=data_collator
)

trainer.train()
```

### 6. Inference (Generate Summaries)
```python
def generate_summary(text):
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=50,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
dialogue = "Mike: How was your day? Lisa: Pretty good..."
summary = generate_summary(dialogue)
print(summary)
```

### 7. Evaluation
```python
from datasets import load_metric

rouge_metric = load_metric("rouge")

# Compute ROUGE scores
results = rouge_metric.compute(
    predictions=[generated_summary],
    references=[reference_summary]
)
print(results)
```

## Model Architecture: PEGASUS

### Key Components
1. **Encoder**: Processes input dialogue
   - Self-attention layers
   - Feed-forward networks
   - Captures context from conversation

2. **Decoder**: Generates summary
   - Cross-attention to encoder
   - Generates tokens sequentially
   - Uses beam search for better quality

3. **Pre-training Objective**: 
   - MLM (Masked Language Modeling)
   - Gap-sentence generation

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_input_length | 1024 | Maximum dialogue length |
| max_target_length | 128 | Maximum summary length |
| num_beams | 4 | Beam search width |
| learning_rate | 2e-5 | Training learning rate |
| batch_size | 8 | Training batch size |
| epochs | 3 | Training epochs |

## ROUGE Metrics for Evaluation

- **ROUGE-1**: Unigram overlap between generated and reference summary
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: ROUGE-L applied to entire summary

```
ROUGE Scores:
  ROUGE-1: 0.42 (Recall), 0.40 (Precision), 0.41 (F1)
  ROUGE-2: 0.19 (Recall), 0.18 (Precision), 0.18 (F1)
  ROUGE-L: 0.38 (Recall), 0.36 (Precision), 0.37 (F1)
```

## Advanced Features

### Batch Processing
```python
dialogues = [dialogue1, dialogue2, dialogue3]
inputs = tokenizer(dialogues, return_tensors="pt", padding=True).to(device)
summary_ids = model.generate(inputs['input_ids'])
summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
```

### Fine-tuning on Custom Data
1. Prepare dataset with 'dialogue' and 'summary' columns
2. Convert to feature format
3. Train using Seq2SeqTrainer
4. Save model for production

### Temperature & Sampling
```python
summary_ids = model.generate(
    inputs,
    max_length=150,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

## Performance Expectations

| Metric | Score |
|--------|-------|
| ROUGE-1 F1 | ~0.40-0.45 |
| ROUGE-2 F1 | ~0.18-0.22 |
| ROUGE-L F1 | ~0.37-0.42 |
| Inference Time | ~0.5-1.0 sec/dialogue |

## Use Cases
- **Customer Service**: Summarize support conversations
- **Meeting Notes**: Auto-generate meeting minutes
- **News Articles**: Create headlines and summaries
- **Legal Documents**: Condense contracts and agreements
- **Research Papers**: Generate abstracts
- **Chat Applications**: Message summarization

## Optimization Techniques

1. **Model Quantization**: Reduce model size
2. **Knowledge Distillation**: Faster smaller models
3. **ONNX Export**: Framework-agnostic deployment
4. **Batch Processing**: Process multiple dialogues at once
5. **Caching**: Cache encoder outputs

## Limitations
- Struggles with very long documents (>1024 tokens)
- May hallucinate or create information not in source
- Requires fine-tuning for domain-specific tasks
- Computational cost for inference

## Future Enhancements
- Multi-document summarization
- Query-focused summarization
- Extractive + abstractive hybrid approach
- Low-resource fine-tuning (LoRA)
- Real-time API deployment
- Mobile-optimized models

## References
- [Google PEGASUS Paper](https://arxiv.org/abs/1912.08777)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [ROUGE Evaluation Metrics](https://en.wikipedia.org/wiki/ROUGE_(metric))

## Author
Data Science & AI Learning Project

---
**Last Updated**: January 2026

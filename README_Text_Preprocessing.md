# Text Preprocessing Notebook

## Overview
This notebook covers common text preprocessing steps for NLP tasks, demonstrated using the IMDB movie reviews dataset. It includes cleaning, normalization, and tokenization techniques required before feature extraction and model training.

## Features
- Lowercasing text
- Removing HTML tags and URLs
- Punctuation handling and removal
- Stopword removal
- Tokenization using NLTK
- Basic exploratory data analysis (EDA)

## Example Steps
```python
import re

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

# Lowercase
# df['review'] = df['review'].str.lower()

# Remove URLs
# def remove_url(text):
#     pattern = re.compile(r'https?://\S+|www\.\S+')
#     return pattern.sub(r'', text)
```

## Dataset
- Source: IMDB Dataset of 50K Movie Reviews (Kaggle)

## Author
**Ranit Pal** - Data Science & AI Learning Project
- GitHub: [@ranit57](https://github.com/ranit57)
- Email: ranitpal57@gmail.com

---
**Last Updated**: January 2026

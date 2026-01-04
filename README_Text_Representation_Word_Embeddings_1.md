# Text Representation and Word Embeddings - Part 1

## Overview
This notebook introduces basic text representation techniques such as Bag-of-Words (BoW), N-grams, and TF-IDF. It demonstrates feature extraction using `CountVectorizer` and how to transform text into numerical vectors for machine learning models.

## Features
- Bag-of-Words representation
- N-gram extraction (bi-grams, tri-grams)
- Vocabulary inspection
- Converting new samples using trained vectorizers

## Example
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,1))
bow = cv.fit_transform(df['text'])
print(cv.vocabulary_)
```

## Author
**Ranit Pal** - Data Science & AI Learning Project
- GitHub: [@ranit57](https://github.com/ranit57)
- Email: ranitpal57@gmail.com

---
**Last Updated**: January 2026

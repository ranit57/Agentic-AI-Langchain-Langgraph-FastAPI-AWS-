# Text Representation and Word Embeddings - Part 2

## Overview
This notebook demonstrates training Word2Vec embeddings using `gensim` and preprocessing text into sentences for Word2Vec training. It shows how to build vocabulary, train the model, and visualize embeddings using PCA.

## Features
- Build corpus from text files
- Sentence tokenization using NLTK
- Word2Vec model training with `gensim`
- Similarity queries and vector extraction
- PCA visualization of embedding space

## Example
```python
from gensim.models import Word2Vec
model = Word2Vec(window=10, min_count=2)
model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
print(model.wv.most_similar('daenerys'))
```

## Author
**Ranit Pal** - Data Science & AI Learning Project
- GitHub: [@ranit57](https://github.com/ranit57)
- Email: ranitpal57@gmail.com

---
**Last Updated**: January 2026

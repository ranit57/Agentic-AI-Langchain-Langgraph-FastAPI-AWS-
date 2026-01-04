# Text Classification using Machine Learning

## Overview
This project demonstrates **Sentiment Analysis** on IMDB movie reviews using Deep Learning (LSTM) and Machine Learning techniques. It preprocesses raw text data, extracts features, and trains a neural network classifier to predict whether a review is positive or negative.

## Dataset
- **Source**: [IMDB Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 movie reviews (in this project, using 10,000 samples)
- **Classes**: Binary classification (Positive/Negative sentiment)
- **Format**: CSV with 'review' and 'sentiment' columns

## Features
- **Data Preprocessing**: HTML tag removal, lowercasing, stopword removal
- **Text Vectorization**: CountVectorizer and TF-IDF transformation
- **Deep Learning Model**: LSTM-based neural network
- **Model Evaluation**: Confusion matrix and accuracy metrics
- **GPU Support**: CUDA acceleration for faster training

## Installation
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn keras tensorflow
```

## Requirements
- Python 3.8+
- NumPy & Pandas (data manipulation)
- Scikit-learn (ML algorithms and preprocessing)
- Keras & TensorFlow (deep learning)
- NLTK (natural language toolkit)
- Matplotlib & Seaborn (visualization)
- CUDA 11.0+ (optional, for GPU acceleration)

## Workflow

### 1. Data Loading and Exploration
```python
import pandas as pd
df = pd.read_csv("IMDB Dataset.csv")
df.head()  # View first rows
df.shape   # Dataset dimensions (10000, 2)
df['sentiment'].value_counts()  # Class distribution
```

### 2. Data Cleaning
- **Remove duplicates**: Eliminate duplicate reviews
- **Check missing values**: Handle NaN/null values
- **Remove HTML tags**: Clean HTML markup from reviews
```python
df.drop_duplicates(inplace=True)
df['review'] = df['review'].apply(remove_tags)
```

### 3. Text Preprocessing
- **Lowercasing**: Convert all text to lowercase
- **Stopword Removal**: Remove common English words (the, is, a, etc.)
- **Tokenization**: Split text into individual tokens
- **HTML Tag Removal**: Strip HTML tags with regex

```python
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text
```

### 4. Feature Extraction
#### Option A: Count Vectorizer
```python
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
X_count = count_vec.fit_transform(df['review'])
```

#### Option B: TF-IDF Vectorizer
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['review'])
```

#### Option C: Keras Tokenizer + Embedding
```python
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
sequences = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(sequences, maxlen=500)
```

### 5. Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 6. Model Architecture (LSTM-based)
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=500),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### 7. Model Training
```python
model.compile(
    optimizer=RMSprop(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(X_test, y_test),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint('best_model.h5')
    ]
)
```

### 8. Model Evaluation
```python
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred.round())
```

## Key Libraries

| Library | Purpose |
|---------|---------|
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **NLTK** | Natural language processing |
| **Scikit-learn** | Machine learning algorithms |
| **Keras** | Deep learning models |
| **Matplotlib/Seaborn** | Data visualization |

## Model Architecture Details

### LSTM (Long Short-Term Memory)
- **Embedding Layer**: Converts words to dense vectors (128 dimensions)
- **SpatialDropout1D**: Prevents overfitting (20% dropout)
- **LSTM Layer**: Captures sequential dependencies (100 units, 20% dropout)
- **Dense Layer**: Fully connected layer (64 units, ReLU activation)
- **Output Layer**: Single neuron with sigmoid (binary classification)

## Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: True Positives, False Positives, True Negatives, False Negatives
- **Precision & Recall**: Trade-off metrics for imbalanced data

## Hyperparameters
- **Vocab Size**: 5000 unique words
- **Max Sequence Length**: 500 tokens
- **Embedding Dimension**: 128
- **LSTM Units**: 100
- **Batch Size**: 128
- **Epochs**: 5 (with early stopping)
- **Dropout Rate**: 0.2

## Expected Performance
- **Training Accuracy**: ~95%+
- **Test Accuracy**: ~88-92%
- **Training Time**: ~10-20 minutes (GPU)

## Tips for Improvement
1. Increase training data size
2. Use pre-trained word embeddings (GloVe, Word2Vec)
3. Implement bidirectional LSTM
4. Use attention mechanisms
5. Hyperparameter tuning (learning rate, batch size, layers)
6. Data augmentation and balancing

## Limitations
- Requires labeled training data
- Computationally intensive for large datasets
- May not capture sarcasm or context-dependent sentiment
- Performance depends on data quality

## Future Enhancements
- Transfer learning with pre-trained models (BERT, RoBERTa)
- Multi-class sentiment analysis (1-5 star ratings)
- Aspect-based sentiment analysis
- Real-time prediction API
- Deployment to production

## References
- [IMDB Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

## Author
**Ranit Pal** - Data Science & AI Learning Project
- GitHub: [@ranit57](https://github.com/ranit57)
- Email: ranitpal57@gmail.com

---
**Last Updated**: January 2026

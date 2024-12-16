# Sentiment Analysis for Movie Reviews

This project implements a sentiment analysis model to classify movie reviews as **positive** or **negative**. The model is built using deep learning techniques, specifically an LSTM (Long Short-Term Memory) network, to process and analyze text data.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)


## Overview

This project uses the **IMDB Movie Reviews Dataset** to perform binary sentiment classification. It preprocesses the review text, tokenizes the text into sequences, and trains an LSTM model for classification.

### Key Features:
- Text preprocessing: Remove special characters, URLs, and stopwords
- Tokenization and padding of text sequences
- LSTM model for sentiment classification
- Visualizations of training and validation accuracy/loss

## Requirements

To run this project, you need the following libraries:

- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `nltk`
## Dataset
The dataset used for this project is the IMDB Movie Reviews Dataset. It consists of 50,000 movie reviews, each labeled as either positive or negative sentiment.

## Preprocessing
The text data undergoes the following preprocessing steps:

* Cleaning: Remove HTML tags, URLs, non-alphanumeric characters, and convert text to lowercase.
* Stopword Removal: Remove commonly used words (like "the", "and") that do not contribute to sentiment.
* Lemmatization: Convert words to their root form (e.g., "running" to "run").
The reviews are then tokenized into sequences of integers, and the sequences are padded to ensure uniform length for training.

## Model Architecture
The model architecture consists of the following layers:

* Embedding Layer: Transforms integer-encoded text sequences into dense vectors.
* LSTM Layer: A type of RNN (Recurrent Neural Network) used to capture long-term dependencies in the text.
* Dense Layers: Fully connected layers used for classification.
* Dropout Layers: Regularization technique to prevent overfitting.

## Training
The model is trained on the movie reviews dataset with the following settings:
* Epochs: 5
* Batch Size: 64
* Optimizer: Adam
* Loss Function: Binary Cross-Entropy
During training, the model is validated using a separate test set to ensure good generalization.
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64
)
```
## Evaluation
After training, the model's performance is evaluated on the test set. The evaluation metrics include:

* Test Accuracy:  0.8802000284194946
* Test Loss: 0.4153500497341156
  
## Usage
To use the trained model for sentiment prediction:
Preprocess your review text as described above.
Tokenize and pad the input sequence.
Use the model to predict sentiment

```python
# Example prediction
review_text = "This movie was amazing!"
review_text = preprocess_review(review_text)  # Preprocessing function
review_sequence = tokenizer.texts_to_sequences([review_text])
padded_review = pad_sequences(review_sequence, maxlen=200)

# Predict sentiment
prediction = model.predict(padded_review)
if prediction > 0.5:
    print("Positive Sentiment")
else:
    print("Negative Sentiment")



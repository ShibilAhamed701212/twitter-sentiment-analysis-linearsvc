# Twitter Sentiment Analysis

This repository contains a simple machine learning pipeline for Twitter Sentiment Analysis. It trains a `LogisticRegression` model using TF-IDF features.

## Setup

1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Dataset

By default, the script looks for `training.1600000.processed.noemoticon.csv` (Kaggle Sentiment140 dataset). If it is not found, the script gracefully falls back to the `nltk.corpus.twitter_samples` dataset to train a sample model.

## Training

Run the training script:
```bash
python train_model.py
```

This will output `trained_model.sav` and `vectorizer.sav` files which can be used for inference.

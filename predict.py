"""Classify tweets with the model saved by train_model.py.

Usage:
    python predict.py "I love this!" "worst day ever"

Only load .sav files you created yourself: pickle can execute code on load.
"""
import argparse
import pickle
import sys

from train_model import MODEL_PATH, VECTORIZER_PATH, stemming

LABELS = {0: 'Negative', 1: 'Positive'}


def load(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict(texts, model, vectorizer):
    features = vectorizer.transform([stemming(t) for t in texts])
    return [LABELS[int(p)] for p in model.predict(features)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('texts', nargs='+', help='tweet text(s) to classify')
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--vectorizer', default=VECTORIZER_PATH)
    args = parser.parse_args(argv)

    try:
        model, vectorizer = load(args.model, args.vectorizer)
    except FileNotFoundError as e:
        print(f"{e.filename} not found. Run `python train_model.py` first.", file=sys.stderr)
        return 1

    for text, label in zip(args.texts, predict(args.texts, model, vectorizer)):
        print(f"{label}\t{text}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

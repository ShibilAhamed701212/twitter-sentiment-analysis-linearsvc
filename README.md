# Twitter Sentiment Analysis (LinearSVC)

A small, script-based machine learning pipeline that classifies tweets as **Positive** or **Negative**. Tweets are cleaned and stemmed with NLTK, turned into TF-IDF unigram + bigram features, and classified with scikit-learn's `LinearSVC`.

## Features

- Trains on the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset (1.6M labelled tweets) when it is present locally.
- Falls back to NLTK's built-in `twitter_samples` corpus (10,000 tweets) so the pipeline runs without a Kaggle account.
- Saves the trained model and vectorizer with `pickle`.
- `predict.py` command-line tool that reloads the saved files and classifies new text with the same preprocessing used in training.
- Unit tests and a GitHub Actions workflow (ruff + pytest).

## Tech stack

Python 3.11, pandas, NLTK (stopwords, Porter stemmer, `twitter_samples`), scikit-learn (`TfidfVectorizer`, `LinearSVC`), pytest, ruff.

## How it works

```
raw tweet
  → keep letters only, lowercase, drop English stopwords, Porter-stem   (train_model.stemming)
  → TfidfVectorizer(ngram_range=(1, 2), max_features=500_000)          (fit on the training split only)
  → LinearSVC(max_iter=2000, dual=True)
  → 0 = Negative, 1 = Positive
```

The data is split 80/20 with stratification (`random_state=2`). Sentiment140 labels `0`/`4` are mapped to `0`/`1`.

## Project structure

| Path | Purpose |
| --- | --- |
| `train_model.py` | Loads data, preprocesses, trains, prints train/test accuracy, saves `trained_model.sav` and `vectorizer.sav`. |
| `predict.py` | CLI that loads the saved model and vectorizer and classifies text. |
| `tests/` | pytest suite (preprocessing, dataset loading, train → save → predict round trip, CLI error path). |
| `Untitled2.ipynb` | Original Colab exploration notebook (Logistic Regression on Sentiment140). Kept for reference. |
| `extracted_code.py` | Raw export of the notebook's cells, including shell commands; not runnable as a script. |
| `.github/workflows/ci.yml` | Runs `ruff check .` and `pytest` on pushes to `main` and on pull requests. |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt for tests and lint
```

The NLTK `stopwords` (and, for the fallback, `twitter_samples`) corpora are downloaded automatically on first run if they are missing. If the download fails, the script stops with NLTK's error instead of continuing.

## Dataset (optional)

`train_model.py` looks for `training.1600000.processed.noemoticon.csv` in the working directory. To get it, install the Kaggle CLI and configure your `kaggle.json` credentials, then:

```bash
pip install kaggle
kaggle datasets download -d kazanova/sentiment140
unzip sentiment140.zip
```

Without the file, training uses NLTK `twitter_samples` instead.

## Usage

Train:

```bash
python train_model.py
```

Classify text with the saved model:

```bash
python predict.py "I love this so much" "this is the worst day ever"
```

`--model` and `--vectorizer` override the default file paths. If the files don't exist, the tool exits with status 1 and asks you to train first.

## Example output

These are real runs on the NLTK `twitter_samples` fallback (Python 3.11, scikit-learn 1.9.1, NLTK 3.10.3):

```text
$ python train_model.py
Kaggle dataset not found. Falling back to NLTK twitter_samples for build...
...
Data shape: (10000, 2)
Applying stemming...
Vectorizing...
Training model...
Accuracy score of the training data :  0.998
Accuracy score of the test data :  0.7615
Saving model and vectorizer...
Model successfully built!

$ python predict.py "I love this so much" "this is the worst day ever" "meh"
Positive	I love this so much
Negative	this is the worst day ever
Negative	meh
```

The LinearSVC pipeline has **not** been evaluated on the full Sentiment140 dataset in this repository, so no accuracy figure is claimed for it. For reference, the notebook's earlier Logistic Regression model reported 0.805 train / 0.777 test accuracy on Sentiment140; that number does not apply to the current model.

## Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests need the NLTK `stopwords` corpus, which is downloaded on first use.

## Known limitations

- **Negations are removed.** NLTK's English stopword list includes words like `not`, `no` and `don't`, so "not good" becomes `good`. This limits accuracy on negated sentences.
- **Only letters are kept.** Emoticons, numbers and punctuation are dropped, and URLs and @handles are split into word fragments (`http`, `co`, username parts) rather than removed.
- **Binary labels only.** There is no neutral class.
- **Overfitting on the small fallback dataset** is visible in the 0.998 train vs 0.7615 test accuracy above.
- **Pickle files are trusted input.** `predict.py` uses `pickle.load`, which can execute code; only load `.sav` files you produced yourself.
- **The notebook differs from the script.** It fits the vectorizer on the full dataset (including the test split) before evaluating, which leaks test data into the features, and some cells were run out of order. `train_model.py` fits the vectorizer on the training split only.
- On the full 1.28M-tweet training split, `LinearSVC(max_iter=2000)` may emit a `ConvergenceWarning`; this was not verified here.
- Dependency versions are not pinned; the versions in the example output above are the ones tested.

## License

No license file is included in this repository.

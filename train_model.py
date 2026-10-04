import os
import pickle
import re

import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

CSV_PATH = 'training.1600000.processed.noemoticon.csv'
MODEL_PATH = 'trained_model.sav'
VECTORIZER_PATH = 'vectorizer.sav'

port_stem = PorterStemmer()
_stop_words = None


def ensure_nltk_resource(name):
    """Download an NLTK corpus only if it is missing, failing loudly on error."""
    try:
        nltk.data.find(f'corpora/{name}')
    except LookupError:
        # raise_on_error makes a failed download stop here instead of
        # surfacing later as a confusing LookupError.
        nltk.download(name, quiet=True, raise_on_error=True)


def get_stop_words():
    global _stop_words
    if _stop_words is None:
        ensure_nltk_resource('stopwords')
        from nltk.corpus import stopwords
        _stop_words = set(stopwords.words('english'))
    return _stop_words


def stemming(content):
    stop_words = get_stop_words()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def load_dataset(csv_path=CSV_PATH):
    if os.path.exists(csv_path):
        print("Loading Kaggle Sentiment140 dataset...")
        col = ['target', 'ids', 'date', 'flag', 'user', 'text']
        twitter_data = pd.read_csv(csv_path, names=col, encoding='ISO-8859-1')
        twitter_data.replace({'target': {4: 1}}, inplace=True)
        return twitter_data

    print("Kaggle dataset not found. Falling back to NLTK twitter_samples for build...")
    ensure_nltk_resource('twitter_samples')
    from nltk.corpus import twitter_samples

    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')

    data = []
    for tweet in pos_tweets:
        data.append({'text': tweet, 'target': 1})
    for tweet in neg_tweets:
        data.append({'text': tweet, 'target': 0})

    return pd.DataFrame(data)


def train(twitter_data):
    """Stem, split, vectorize and fit. Returns (model, vectorizer, train_acc, test_acc)."""
    print("Applying stemming...")
    twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

    X = twitter_data['stemmed_content'].values
    Y = twitter_data['target'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    print("Vectorizing...")
    # Using bigrams and limiting features to prevent memory issues but capture more context
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print("Training model...")
    # LinearSVC typically yields higher accuracy on sparse text data than standard Logistic Regression
    model = LinearSVC(max_iter=2000, random_state=2, dual=True)
    model.fit(X_train, Y_train)

    training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
    test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))
    return model, vectorizer, training_data_accuracy, test_data_accuracy


def save(model, vectorizer, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)


def main():
    twitter_data = load_dataset()
    print(twitter_data.head())
    print("Data shape:", twitter_data.shape)

    model, vectorizer, training_data_accuracy, test_data_accuracy = train(twitter_data)
    print('Accuracy score of the training data : ', training_data_accuracy)
    print('Accuracy score of the test data : ', test_data_accuracy)

    print("Saving model and vectorizer...")
    save(model, vectorizer)

    print("Model successfully built!")


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

nltk.download('stopwords')

csv_path = 'training.1600000.processed.noemoticon.csv'

if os.path.exists(csv_path):
    print("Loading Kaggle Sentiment140 dataset...")
    col = ['target', 'ids', 'date', 'flag', 'user', 'text']
    twitter_data = pd.read_csv(csv_path, names=col, encoding='ISO-8859-1')
    twitter_data.replace({'target':{4:1}}, inplace=True)
else:
    print("Kaggle dataset not found. Falling back to NLTK twitter_samples for build...")
    nltk.download('twitter_samples')
    from nltk.corpus import twitter_samples
    
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    
    data = []
    for tweet in pos_tweets:
        data.append({'text': tweet, 'target': 1})
    for tweet in neg_tweets:
        data.append({'text': tweet, 'target': 0})
        
    twitter_data = pd.DataFrame(data)

print(twitter_data.head())
print("Data shape:", twitter_data.shape)

port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

print("Applying stemming...")
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

print("Vectorizing...")
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data : ', test_data_accuracy)

print("Saving model and vectorizer...")
with open('trained_model.sav', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.sav', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model successfully built!")

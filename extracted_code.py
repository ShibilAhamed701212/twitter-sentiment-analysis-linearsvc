pip install kaggle



!kaggle datasets download -d kazanova/sentiment140


!unzip -o sentiment140.zip


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import nltk
nltk.download('stopwords')


print(stopwords.words('english'))


twitter_data.shape



col = ['target', 'ids', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv',names=col, encoding='ISO-8859-1')


twitter_data.head()


twitter_data.isnull().sum()


twitter_data['target'].value_counts()


twitter_data.replace({'target':{4:1}},inplace=True)


port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)


X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


print(X.shape, X_train.shape, X_test.shape)


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


print('Accuracy score of the training data : ', training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


print('Accuracy score of the test data : ', test_data_accuracy)


import pickle


filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))


X_new = X_test[200]
print(Y_test[200])
prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('Negative')

else:
  print('Positive')



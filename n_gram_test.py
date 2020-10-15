import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob


"""*********************************************************
                    Loading Data
*********************************************************"""

np.set_printoptions(precision=2)
pd.set_option("display.max_columns", 10)

data = pd.read_csv("sentiment.csv")
data["sentiment"] = data["sentiment"] == "pos"
data["sentiment"] = data["sentiment"].astype("bool")

data_train, data_test = train_test_split(data, test_size=0.3)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]

"""*********************************************************
                    BOW as input
*********************************************************"""
# computing Bag of Words
vectorizer = CountVectorizer()
vectorizer.fit(data_train["text"])
x_train, x_test = vectorizer.transform(data_train["text"]), \
                  vectorizer.transform(data_test["text"])

# 9. Fitting BOW into LR with CV
print("\n LogisticRegressionCV [BOW features]")
model = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000).fit(x_train,y_train)

print("\nModel score [BOW features] and [CV]")
print("training set score: ", model.score(x_train, y_train))
print("testing set score:  ", model.score(x_test, y_test))


"""*********************************************************
                    BOW n-grams
*********************************************************"""


vectorizer = CountVectorizer(ngram_range=(1,2))
vectorizer.fit(data_train["text"])
x_train, x_test = vectorizer.transform(data_train["text"]), \
                  vectorizer.transform(data_test["text"])

# 9. Fitting BOW into LR
print("\n LogisticRegressionCV [BOW n-gram; n=2]")
model = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000).fit(x_train,y_train)

print("\nModel score [BOW n-gram; n=2] and [CV]")
print("training set score: ", model.score(x_train, y_train))
print("testing set score:  ", model.score(x_test, y_test))

"""
from:  https://stackoverflow.com/questions/51621307/bag-of-words-bow-vs-n-gram-sklearn-countvectorizer-text-documents-classifi

The main advantages of ngrams over BOW i to take into account the sequence of words. 
For instance, in the sentences:

"I love apples but I hate grapes"
"I love grapes but I hate apples"

same bow, but different n-gram seq for n=2
"""
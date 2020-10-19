import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


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
                    BOW as input (Unigrams)
*********************************************************"""
# computing Bag of Words
vectorizer = CountVectorizer()
vectorizer.fit(data_train["text"])
x_train = vectorizer.transform(data_train["text"])
x_test = vectorizer.transform(data_test["text"])

# 9. Fitting BOW into LR with strong regularization
print("\n LogisticRegression [BOW features], reg[C=0.01]")
model = LogisticRegression(C=0.01, random_state=0, max_iter=1000).fit(x_train, y_train)

print("\nModel score [BOW features]")
print("training set score: ", model.score(x_train, y_train))
print("testing set score:  ", model.score(x_test, y_test))


"""*********************************************************
                    BOW n-grams n = 1, 2
*********************************************************"""


vectorizer = CountVectorizer(ngram_range=(1, 2))
vectorizer.fit(data_train["text"])
x_train = vectorizer.transform(data_train["text"])
x_test = vectorizer.transform(data_test["text"])

# 9. Fitting BOW into Logistic regression
print("\n LogisticRegression [BOW n-gram; n <=2], reg[C=0.01]")
model = LogisticRegression(C=0.01, random_state=0, max_iter=1000).fit(x_train, y_train)

print("\n Model score [BOW n-gram; n <= 2]")
print("training set score: ", model.score(x_train, y_train))
print("testing set score:  ", model.score(x_test, y_test))

"""*********************************************************
                    BOW n-grams n = 1, 2, 3
*********************************************************"""

vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorizer.fit(data_train["text"])
x_train = vectorizer.transform(data_train["text"])
x_test = vectorizer.transform(data_test["text"])

# 9. Fitting BOW into Logistic regression
print("\n LogisticRegression [BOW n-gram; n <= 3], reg[C=0.01]")
model = LogisticRegression(C=0.01, random_state=0, max_iter=1000).fit(x_train, y_train)

print("\n Model score [BOW n-gram; n <=3]")
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

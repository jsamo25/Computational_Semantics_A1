import pandas as pd
from pdb import set_trace

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

""" PART I: Bag of words representations as features """
#TODO: read --> https://en.wikipedia.org/wiki/Bag-of-words_model
#65 import data  & turn sentiment column into a boolean representation.
data = pd.read_csv("sentiment.csv")
data["sentiment"] = data["sentiment"].astype("bool") #assign boolean values to sentiment column

#66,67,68: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#X_train, X_test, y_train, y_test = train_test_split(CountVectorizer().fit_transform(data["text"]), data["sentiment"], test_size=0.2, random_state= 1)

#66 data_train <- 80% of data, and the remaining to 20%
data_train, data_test = train_test_split(data, test_size=0.2)
#67 assign the corresponding labels to the output vector
y_train, y_test = data_train["sentiment"], data_test["sentiment"]

#68 assign the corresponding values to the feature vector , following the example of https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
vectorizer = CountVectorizer()
vectorizer.fit(data_train["text"])
#69 print first 50 features
print("feature name", vectorizer.get_feature_names()[:50])
#70 transform the text column
x_train, x_test = vectorizer.transform(data_train["text"]),vectorizer.transform(data_test["text"])

print(x_train.shape)
print(type(x_train))
print(x_train.toarray()) #FIXME: hard to display all (use sentiment10.csv)
print(vectorizer.vocabulary_)

""" PART J: Logistic regression model based on bag-of-words representation """

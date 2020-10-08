import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

""" PART I: Bag of words representations as features """
#65 import data  & turn sentiment column into a boolean representation.
data = pd.read_csv("sentiment.csv")
data["sentiment"] = data["sentiment"].astype("bool") #assign boolean values to sentiment column

#66,67,68: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#FIXME: remove random_state parameter to allow the method to split the data set randomly
X_train, X_test, y_train, y_test = train_test_split(CountVectorizer().fit_transform(data["text"]), data["sentiment"], test_size=0.2, random_state= 1)

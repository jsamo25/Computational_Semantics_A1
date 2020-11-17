import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from matplotlib.colors import Normalize

"""*********************************************************
                    Loading Data
*********************************************************"""

data = pd.read_csv("data/sentiment.csv")
data["sentiment"] = data["sentiment"] == "pos"
data["sentiment"] = data["sentiment"].astype("bool")

data_train, data_test = train_test_split(data, test_size=0.25)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]

# features
vectorizer = CountVectorizer()
vectorizer.fit(data_train["text"])
x_train, x_test = vectorizer.transform(data_train["text"]), \
                  vectorizer.transform(data_test["text"])


"""*********************************************************
                    Testing SVM
*********************************************************"""

model = svm.SVC(random_state=0, gamma="auto").fit(x_train,y_train)

print("\nModel score [BOW features] ")
print("training set score: ", model.score(x_train, y_train))
print("testing set score:  ", model.score(x_test, y_test))
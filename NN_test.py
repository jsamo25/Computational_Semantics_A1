import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

"""*********************************************************
                    Loading Data
*********************************************************"""

data = pd.read_csv("data/sentiment.csv")
data["sentiment"] = data["sentiment"] == "pos"
data["sentiment"] = data["sentiment"].astype("bool")

data_train, data_test = train_test_split(data, test_size=0.4)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]

# features
vectorizer = CountVectorizer()
vectorizer.fit(data_train["text"])
x_train, x_test = vectorizer.transform(data_train["text"]), \
                  vectorizer.transform(data_test["text"])

"""*********************************************************
                Using Multilayer perceptron
*********************************************************"""

model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1).fit(x_train,y_train)
y_pred = model.predict(x_test)

print("\nModel MLP score [BOW features] ")
print("training set score: ", model.score(x_train, y_train))
print("testing set score:  ", model.score(x_test, y_test))
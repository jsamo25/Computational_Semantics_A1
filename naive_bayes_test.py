import pandas as pd
import numpy as np
import nltk

from sklearn.naive_bayes import GaussianNB
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

positive_lexicons = ["admire", "amazing", "assure", "celebration", "charm", "eager", "enthusiastic", "excellent",
                     "fancy", "fantastic", "frolic", "graceful", "happy", "joy", "luck", "majesty", "mercy", "nice",
                     "patience", "perfect", "proud", "rejoice", "relief", "respect", "satisfactorily", "sensational",
                     "super", "terrific", "thank", "vivid", "wise", "wonderful", "zest", "good", "great"]
negative_lexicons = ["abominable", "anger", "anxious", "bad", "catastrophe", "cheap", "complaint", "condescending",
                     "deceit","defective", "disappointment", "embarrass", "fake", "fear", "filthy", "fool", "guilt",
                     "hate", "idiot", "inflict", "lazy", "miserable", "mourn", "nervous", "objection", "pest", "plot",
                     "reject", "scream", "silly", "terrible", "unfriendly","vile", "wicked"]

# 2. #Initial text analysis and basic statistics of the dataset.
data["sentences"] = data["text"].apply(nltk.tokenize.sent_tokenize)
data["tokens"] = data["text"].apply(nltk.tokenize.word_tokenize)
data["n_characters"] = data["text"].apply(lambda x: len(x))
data["n_tokens"] = data["tokens"].apply(lambda x: len(x))
data["n_sentences"] = data["sentences"].apply(lambda x: len(x))

def count_positive_lexicons(tokens):
    return sum(lexicon in positive_lexicons for lexicon in tokens)
data["n_positive_lex"] = data["tokens"].apply(count_positive_lexicons)

def count_negative_lexicons(tokens):
    return sum(lexicon in negative_lexicons for lexicon in tokens)
data["n_negative_lex"] = data["tokens"].apply(count_negative_lexicons)

"""*********************************************************
                    Train/Test split
*********************************************************"""

data_train, data_test = train_test_split(data, test_size=0.3)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]

# 6. Data split, training/test and evaluation function.
data_train, data_test = train_test_split(data, test_size=0.3)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]
x_train, x_test = (
    data_train[["n_characters", "n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]],
     data_test[["n_characters", "n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]]
)

"""*********************************************************
                    Gaussian Naive Bayes
*********************************************************"""

# 9. Fitting features into GNB w
print("\n Gaussian Naive Bayes [hand-chosen features]")
model = GaussianNB().fit(x_train,y_train)
y_pred = model.predict(x_test)

print("\nModel score [Hand-chosen features]")
print("training set score: ", model.score(x_train, y_train))
print("testing set score:  ", model.score(x_test, y_test))

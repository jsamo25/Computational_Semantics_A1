import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import plot_confusion_matrix
from pdb import set_trace

"""Importing the dataset + target column boolean transformation """
data = pd.read_csv("sentiment10.csv")
data["sentiment"] = data["sentiment"] == "pos"
data["sentiment"] = data["sentiment"].astype("bool") #assign boolean values to sentiment column

"""PART J. Basic task and data exploration [max 2 points]"""
# 2. #Report some basic statistics of the dataset (means, standard deviations). Also
# report the average number of words per review and the average number of sentences per review.

data['n_characters'] = data['text'].apply(lambda x: len(x))
data["sentences"] = data["text"].apply(nltk.tokenize.sent_tokenize)
data["n_sentences"] = data["sentences"].apply(lambda x: len(x))
data['tokens'] = data['text'].apply(nltk.tokenize.word_tokenize)
data['n_tokens'] = data['tokens'].apply(lambda x: len(x))

print(data.describe())

# 3. Histogram of the rating column
plt.hist(data["rating"])
plt.title("Rating Histogram")
plt.grid(True)
#plt.show()

# 4. av. words/sentences for pos/neg reviews
pos_rev_word_avg = (data["n_characters"].groupby(data["sentiment"]==True)).mean()
neg_rev_word_avg = (data["n_characters"].groupby(data["sentiment"]==False)).mean()
pos_rev_sent_avg = (data["n_sentences"].groupby(data["sentiment"]==True)).mean()
neg_rev_sent_avg = (data["n_sentences"].groupby(data["sentiment"]==False)).mean()

""" PART K. Logistic regression with hand-chosen features"""
#5 feature selection: # characters, # words, # sentences, # possitive words, # negative words (using ["rate"] is cheating)
#TODO Add more lexicons to improve precision
#TODO: perhaps taking the lexicons from a separate CSV file, dictionary, or similar.
positive_lexicons = ["admire", "amazing", "assure", "celebration", "charm", "eager", "enthusiastic", "excellent",
                     "fancy", "fantastic", "frolic", "graceful", "happy", "joy", "luck", "majesty", "mercy", "nice",
                     "patience", "perfect", "proud", "rejoice", "relief", "respect", "satisfactorily", "sensational",
                     "super", "terrific", "thank", "vivid", "wise", "wonderful", "zest", "good", "great"]
negative_lexicons = ["abominable", "anger", "anxious", "bad", "catastrophe", "cheap", "complaint", "condescending",
                     "deceit","defective", "disappointment", "embarrass", "fake", "fear", "filthy", "fool", "guilt",
                     "hate", "idiot", "inflict", "lazy", "miserable", "mourn", "nervous", "objection", "pest", "plot",
                     "reject", "scream", "silly", "terrible", "unfriendly","vile", "wicked"]

def positive_lex(tokens):
    return sum(lexicon in positive_lexicons for lexicon in tokens)
data['n_positive_lex'] = data['tokens'].apply(positive_lex)
#print(data["n_positive_lex"])

def negative_lex(tokens):
    return sum(lexicon in negative_lexicons for lexicon in tokens)
data['n_negative_lex'] = data['tokens'].apply(negative_lex)
#print(data["n_negative_lex"])

#6 Logistic regression

data_train, data_test = train_test_split(data, test_size=0.2)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]
x_train, x_test= data_train[["n_characters","n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]],data_test[["n_characters","n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]]
model = LogisticRegression(max_iter=1000).fit(x_train,y_train)

print("\n logistic regression score (train):", model.score(x_train, y_train))
print("\n logistic regression score (test) :", model.score(x_test, y_test))

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, x_test, y_test,
                                 display_labels=["Positive","Negative"],
                                 #cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
#plt.show()


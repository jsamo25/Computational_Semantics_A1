import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from pdb import set_trace

"""Importing the dataset + target column boolean transformation """
data = pd.read_csv("sentiment10.csv")
data["sentiment"] = data["sentiment"] == "pos"
data["sentiment"] = data["sentiment"].astype("bool") #assign boolean values to sentiment column

"""PART J. Basic task and data explporation [max 2 points]"""
# 2. #Report some basic statistics of the dataset (means, standard deviations). Also
# report the average number of words per review and the average number of sentences per review.

data['n_characters'] = data['text'].apply(lambda x: len(x))
data["sentences"] = data["text"].apply(nltk.tokenize.sent_tokenize)
data["n_sentences"] = data["sentences"].apply(lambda x: len(x))
data['tokens'] = data['text'].apply(nltk.tokenize.word_tokenize)
data['n_tokens'] = data['tokens'].apply(lambda x: len(x))
# 3. Histogram of the rating column
plt.hist(data["rating"])
plt.title("Rating Histogram")
plt.grid(True)
plt.show()
# 4. av. words/sentences for pos/neg reviews
pos_rev_word_avg = (data["n_characters"].groupby(data["sentiment"]==True)).mean()
neg_rev_word_avg = (data["n_characters"].groupby(data["sentiment"]==False)).mean()
pos_rev_sent_avg = (data["n_sentences"].groupby(data["sentiment"]==True)).mean()
neg_rev_sent_avg = (data["n_sentences"].groupby(data["sentiment"]==False)).mean()


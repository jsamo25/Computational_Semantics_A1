import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

""" Getting Started """
pd.set_option("display.max_columns", 99)
data = pd.read_csv("sentiment10.csv")
#data = pd.read_csv("sentiment.csv")
print(data["text"])

def part_b():
    #10, 11, 12, 13
    print(data.shape)
    print(data[:3])
    print(data["sentiment"])
    print(data["sentiment"].unique())

    #15, 16
    #print(data["sentiment"]=="pos") #checks if each row matches the "pos" value
    data["sentiment"] = data["sentiment"] == "pos"
    #print(data)
    data["sentiment"] = data["sentiment"].astype("bool") #assign boolean values to sentiment column
    #print(data)

    #17, 18, 19
    data['n_characters'] = data['text'].apply(lambda x: len(x))
    #data['n_characters'] = data['text'].str.len()
    data['tokens'] = data['text'].apply(nltk.tokenize.word_tokenize)
    data['n_tokens'] = data['tokens'].apply(lambda x: len(x))
    #data['n_tokens'] = data['tokens'].str.len()

    #print(data)
    #21, mean, standard deviation
    print("Mean Values\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].mean())
    print("\n Standard deviation Values\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].std())
    #print("Data Statistics", "\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].describe())

    #22 Histogram of the rating column fo the DataFrame sentiment.
    plt.hist(data["rating"])
    print("Data Description\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].describe())
    plt.grid(True)
    plt.show()
def part_c():
    #25The use of features may help to create tendencies, make predictions
    #26 Times that appear the word bad, Times that appear the word good, capitalized words?, exclamation marks?
    #27 selected feature: Times that appear the word "good"

    def n_good(text):
        computed_feature = text.count('good')
        return computed_feature

    data['n_good'] = data['text'].apply(n_good)
    print(data["n_good"])

part_c()
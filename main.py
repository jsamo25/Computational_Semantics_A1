import pandas as pd
import nltk

""" Getting Started """
pd.set_option("display.max_columns", 99)
data = pd.read_csv("sentiment10.csv")


def part_b():
    #10, 11, 12, 13
    print(data.shape)
    # print(data[:3])
    # print(data["sentiment"])
    # print(data["sentiment"].unique())

    #15, 16
    #print(data["sentiment"]=="pos") #checks if each row matches the "pos" value
    #assign a boolean value to a pandas column
    data["sentiment"] = data["sentiment"] == "pos"
    #print(data)
    data["sentiment"] = data["sentiment"].astype("bool")
    #print(data)

    #17, 18
    data['tokens'] = data['text'].apply(nltk.tokenize.word_tokenize)
    print(data)
    # data["tookens"] = data["text"].apply(...)

part_b()
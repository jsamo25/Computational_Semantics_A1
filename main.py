import pandas as pd
import nltk

""" Getting Started """
pd.set_option("display.max_columns", 99)
data = pd.read_csv("sentiment10.csv")


def part_b():
    #10, 11, 12, 13
    #print(data.shape)
    # print(data[:3])
    # print(data["sentiment"])
    # print(data["sentiment"].unique())

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
part_b()
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import random

""" PART A: Getting started """
# TODO: make use of a TRIAL variable to switch between the different data sets (Part B).
pd.set_option("display.max_columns", 99)
data = pd.read_csv("sentiment10.csv")
#data = pd.read_csv("sentiment.csv")
#print(data["text"])

""" PART B: Loading and inspecting the data"""
#TODO: create better functions, maybe a menu?
#FIXME: change commented prints before submitting
#10, 11, 12, 13
# print(data.shape)
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

#FIXME: change commented prints before submitting
#21, mean, standard deviation
# print("Mean Values\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].mean())
# print("\n Standard deviation Values\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].std())
#print("Data Statistics", "\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].describe())

#22 Histogram of the rating column fo the DataFrame sentiment.
# plt.hist(data["rating"])
# print("Data Description\n",data[["rating", "sentiment", "n_characters", "n_tokens"]].describe())
# plt.grid(True)
# plt.show()

""" PART C: Compute features for sentiment classification"""
#25The use of features may help to create functions, and later make predictions
#26 Times that appear the word bad, Times that appear the word good, capitalized words?, exclamation marks?
#27 selected feature: Times that appear the word "good"
#TODO: Find a better feature, "good" or "bad" wont appear all the time, default neutral? (what about negations? i.e not good, not bad. And Sarcasm?)

# def n_good(text):
#     computed_feature = text.count('good')
#     return computed_feature
#
# data['n_good'] = data['text'].apply(n_good)
# print(data["n_good"])

#29 Number of positive & Number of negative words based on polarity lexicon. Figure 21.3 https://web.stanford.edu/~jurafsky/slp3/21.pdf
#TODO: add more Lexicons to improve precision
#FIXME lemmatized tokens will improve the performance?
positive_lexicons = ["admire", "amazing", "assure", "celebration", "charm", "eager", "enthusiastic", "excellent",
                     "fancy", "fantastic", "frolic", "graceful", "happy", "joy", "luck", "majesty", "mercy", "nice",
                     "patience", "perfect", "proud", "rejoice", "relief", "respect", "satisfactorily", "sensational",
                     "super", "terrific", "thank", "vivid", "wise", "wonderful", "zest", "good", "great"]
negative_lexicons = ["abominable", "anger", "anxious", "bad", "catastrophe", "cheap", "complaint", "condescending",
                     "deceit","defective", "disappointment", "embarrass", "fake", "fear", "filthy", "fool", "guilt",
                     "hate", "idiot", "inflict", "lazy", "miserable", "mourn", "nervous", "objection", "pest", "plot",
                     "reject", "scream", "silly", "terrible", "unfriendly","vile", "wicked"]

def positive_lex_n(text):
    computed_feature = sum(lexicon in positive_lexicons for lexicon in text)
    return computed_feature

data['positive_lex_n'] = data['tokens'].apply(positive_lex_n)
print(data["positive_lex_n"])

def negative_lex_n(text):
    computed_feature = sum(lexicon in negative_lexicons for lexicon in text)
    return computed_feature

data['negative_lex_n'] = data['tokens'].apply(negative_lex_n)
print(data["negative_lex_n"])

""" PART D: Code Cleaning: current status: messy"""

""" PART E: Generate predictions based on the features """
#35, 36

def rule_1(feature):
    if feature > 2:
        prediction = True
    else:
        prediction = False
    return prediction

data["predicted_by_rule_1"] = data["positive_lex_n"].apply(rule_1)
#print(data["predicted_by_rule_1"])

def rule_2(feature1, feature2):
    if feature1 > feature2:
        prediction = True
    else:
        prediction = False
    return prediction

data["predicted_by_rule_2"] = data[["positive_lex_n", "negative_lex_n"]].apply(lambda x: rule_2(*x), axis=1)
#print(data["predicted_by_rule_2"])

def rule_3(positive, negative):
    if positive > 2* negative:
        computed_prediction = True
    elif negative > 2* positive:
        computed_prediction = False
    else:
        computed_prediction = random.choice([True, False])
        #TODO: add bias based on... !!
    return computed_prediction

data["predicted_by_rule_3"] = data[["positive_lex_n", "negative_lex_n"]].apply(lambda x: rule_3(*x), axis=1)
print(data["predicted_by_rule_3"])

#39 Baseline predictions

def always_true(text):
    return True
data["baseline_pos"] = data["text"].apply(always_true)
print(data["baseline_pos"])

def always_false(text):
    return False
data["baseline_neg"] = data["text"].apply(always_false)
print(data["baseline_neg"])

def always_random(text):
    list= (True, False)
    prediction = np.random.choice(list, 1, p = [0.75, 0.25])
    return prediction[0]

data["baseline_ran"] = data["text"].apply(always_random)
print(data["baseline_ran"])
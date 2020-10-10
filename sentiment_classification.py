import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from pdb import set_trace

"""*********************************************************
    Import dataset and extra files for sentiment analysis 
*********************************************************"""


data = pd.read_csv("sentiment.csv")
data["sentiment"] = data["sentiment"] == "pos"
data["sentiment"] = data["sentiment"].astype("bool")  # assign boolean values to sentiment column

positive_lexicons = ["admire", "amazing", "assure", "celebration", "charm", "eager", "enthusiastic", "excellent",
                     "fancy", "fantastic", "frolic", "graceful", "happy", "joy", "luck", "majesty", "mercy", "nice",
                     "patience", "perfect", "proud", "rejoice", "relief", "respect", "satisfactorily", "sensational",
                     "super", "terrific", "thank", "vivid", "wise", "wonderful", "zest", "good", "great"]
negative_lexicons = ["abominable", "anger", "anxious", "bad", "catastrophe", "cheap", "complaint", "condescending",
                     "deceit","defective", "disappointment", "embarrass", "fake", "fear", "filthy", "fool", "guilt",
                     "hate", "idiot", "inflict", "lazy", "miserable", "mourn", "nervous", "objection", "pest", "plot",
                     "reject", "scream", "silly", "terrible", "unfriendly","vile", "wicked"]

#A bigger list is provided from https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
#an 5% in accuracy is gained, however the processing time increases significantly
# positive_lexicons = pd.read_csv("positive-words.csv").transpose().values[0]
# negative_lexicons = pd.read_csv("negative-words.csv").transpose().values[0]

"""*********************************************************
    PART J. Basic tasks and data exploration 
*********************************************************"""
# 2. #Initial text analysis and basic statistics of the dataset.

data["sentences"] = data["text"].apply(nltk.tokenize.sent_tokenize)
data["tokens"]    = data["text"].apply(nltk.tokenize.word_tokenize)
data["n_characters"] = data["text"].apply(lambda x: len(x))
data["n_tokens"]     = data["tokens"].apply(lambda x: len(x))
data["n_sentences"]  = data["sentences"].apply(lambda x: len(x))

def count_positive_lexicons(tokens):
    return sum(lexicon in positive_lexicons for lexicon in tokens)
data["n_positive_lex"] = data["tokens"].apply(count_positive_lexicons)

def count_negative_lexicons(tokens):
    return sum(lexicon in negative_lexicons for lexicon in tokens)
data["n_negative_lex"] = data["tokens"].apply(count_negative_lexicons)

print("\n Basic data statistics \n", data.describe())

# 3. Histogram of the rating column
plt.hist(data["rating"])
plt.title("Rating Histogram")
plt.grid(True)
plt.show()

# 4. av. words/sentences for pos/neg reviews
sent_avg_per_sentiment = (data["n_sentences"].groupby(data["sentiment"])).mean()
word_avg_per_sentiment = (data["n_tokens"].groupby(data["sentiment"])).mean()

print("sentences average per sentiment\n", sent_avg_per_sentiment)
print("word average per sentiment\n", word_avg_per_sentiment)

"""*********************************************************
    PART K. Logistic regression with hand-chosen features 
*********************************************************"""
#TODO: using [rating] felt a bit of cheating so I removed that from my list.
print("Selected features:"
                        "\n [n_characters]"
                        "\n [n_tokens]"
                        "\n [n_sentences]"
                        "\n [n_positive_lex]"
                        "\n [n_negative_lex]")
# 6 Logistic regression

data_train, data_test = train_test_split(data, test_size=0.2)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]
x_train, x_test = (
                  data_train[["n_characters", "n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]],
                  data_test[["n_characters", "n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]],)

model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
y_pred = model.predict(x_test)

#7 Evaluation of Model - based on: https://towardsdatascience.com/demystifying-confusion-matrix-confusion-9e82201592fd
def plot_confusion_matrix(cm, classes,normalize=False,title="Confusion matrix",cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("Golden label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=["True", "False"],
                      title="Confusion matrix, without normalization",)
plt.show()

# extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(
    "\n True Negatives: ", tn,
    "\n True Positives: ", tp,
    "\n False Positives: ", fp,
    "\n False Negatives: ", fn
)

# Performance metrics when manual features are used
accuracy  = (tn + tp) * 100 / (tp + tn + fp + fn)
precision = tp * 100 / (tp + fp)
recall    = tp * 100 / (tp + fn)
f1_score  = (2 * precision * recall) / (precision + recall)

print("\n Evaluation metrics:",
      "\n Accuracy: ", accuracy, "%",
      "\n Precision: ", precision, "%",
      "\n Recall: ", recall, "%",
      "\n F1 Score: ", f1_score, "%",
)

"""****************************************************************
    Part L. Logistic regression with bad of words representations 
*****************************************************************"""

vectorizer = CountVectorizer()
vectorizer.fit(data_train["text"])
x_train, x_test = vectorizer.transform(data_train["text"]), \
                  vectorizer.transform(data_test["text"])

model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
y_pred = model.predict(x_test)

# extracting true_positives, false_positives, true_negatives, false_negatives
print("\n Using BOW as feature input")
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=["True", "False"],
                      title="Confusion matrix, without normalization",)
plt.show()

# extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(
    "\n True Negatives: ", tn,
    "\n True Positives: ", tp,
    "\n False Positives: ", fp,
    "\n False Negatives: ", fn
)

# Performance metrics when BOW feature is used
accuracy  = (tn + tp) * 100 / (tp + tn + fp + fn)
precision = tp * 100 / (tp + fp)
recall    = tp * 100 / (tp + fn)
f1_score  = (2 * precision * recall) / (precision + recall)

print("\n Evaluation metrics:",
      "\n Accuracy: ", accuracy, "%",
      "\n Precision: ", precision, "%",
      "\n Recall: ", recall, "%",
      "\n F1 Score: ", f1_score, "%",
)
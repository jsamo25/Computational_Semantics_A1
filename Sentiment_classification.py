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

""" Import dataset and extra files for sentiment analysis"""

data = pd.read_csv("sentiment.csv")
data["sentiment"] = data["sentiment"] == "pos"
data["sentiment"] = data["sentiment"].astype("bool") #assign boolean values to sentiment column

# positive_lexicons = pd.read_csv("positive-words.csv").transpose().values[0]
# negative_lexicons = pd.read_csv("negative-words.csv").transpose().values[0]

positive_lexicons = ["admire", "amazing", "assure", "celebration", "charm", "eager", "enthusiastic", "excellent",
                     "fancy", "fantastic", "frolic", "graceful", "happy", "joy", "luck", "majesty", "mercy", "nice",
                     "patience", "perfect", "proud", "rejoice", "relief", "respect", "satisfactorily", "sensational",
                     "super", "terrific", "thank", "vivid", "wise", "wonderful", "zest", "good", "great"]
negative_lexicons = ["abominable", "anger", "anxious", "bad", "catastrophe", "cheap", "complaint", "condescending",
                     "deceit","defective", "disappointment", "embarrass", "fake", "fear", "filthy", "fool", "guilt",
                     "hate", "idiot", "inflict", "lazy", "miserable", "mourn", "nervous", "objection", "pest", "plot",
                     "reject", "scream", "silly", "terrible", "unfriendly","vile", "wicked"]

"""
    PART J. Basic task and data exploration
"""
# 2. #Initial text analysis and basic statistics of the dataset.

data["sentences"]    = data["text"].apply(nltk.tokenize.sent_tokenize)
data['tokens']       = data['text'].apply(nltk.tokenize.word_tokenize)
data['n_characters'] = data['text'].apply(lambda x: len(x))
data["n_sentences"]  = data["sentences"].apply(lambda x: len(x))
data['n_tokens']     = data['tokens'].apply(lambda x: len(x))

def count_positive_lexicons(tokens):
    return sum(lexicon in positive_lexicons for lexicon in tokens)
data['n_positive_lex'] = data['tokens'].apply(count_positive_lexicons)
print(data["n_positive_lex"])

def count_negative_lexicons(tokens):
    return sum(lexicon in negative_lexicons for lexicon in tokens)
data['n_negative_lex'] = data['tokens'].apply(count_negative_lexicons)
print(data["n_negative_lex"])
print(data.describe())

# 3. Histogram of the rating column
plt.hist(data["rating"])
plt.title("Rating Histogram")
plt.grid(True)
plt.show()

# 4. av. words/sentences for pos/neg reviews
positive_word_avg = (data["n_characters"].groupby(data["sentiment"]==True)).mean()
negative_word_avg = (data["n_characters"].groupby(data["sentiment"]==False)).mean()
positive_sent_avg = (data["n_sentences"].groupby(data["sentiment"]==True)).mean()
negative_sent_avg = (data["n_sentences"].groupby(data["sentiment"]==False)).mean()

""" PART K. Logistic regression with hand-chosen features"""
#5 feature selection: # characters, # words, # sentences, # possitive words, # negative words



#6 Logistic regression

data_train, data_test = train_test_split(data, test_size=0.2)
y_train, y_test = data_train["sentiment"], \
                   data_test["sentiment"]
x_train, x_test= data_train[["n_characters","n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]],\
                  data_test[["n_characters","n_tokens", "n_sentences", "n_positive_lex", "n_negative_lex"]]
model = LogisticRegression(max_iter=1000).fit(x_train,y_train)
y_pred = model.predict(x_test)

#Evaluation of Model - Confusion Matrix Plot based on: https://towardsdatascience.com/demystifying-confusion-matrix-confusion-9e82201592fd

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['True','False'],
                      title='Confusion matrix, without normalization')

plt.show()

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#Accuracy
accuracy= (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy:",accuracy)
#Precision
precision= (tp/(tp+fp))*100
print("Precision:",precision)
#Recall
recall= (tp/(tp+fn))*100
print("Recall: ",recall)
#F1 Score
print("F1 score",(2*precision*recall)/(precision + recall))

"""Part L. Logistic regression with bad of words representations"""
vectorizer = CountVectorizer()
vectorizer.fit(data_train["text"])
x_train, x_test = vectorizer.transform(data_train["text"]),vectorizer.transform(data_test["text"])

model = LogisticRegression(max_iter=1000).fit(x_train,y_train)
y_pred = model.predict(x_test)

#extracting true_positives, false_positives, true_negatives, false_negatives
print("\n Accuracy indicators when using BOW as features")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#Accuracy
accuracy= (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy:",accuracy)
#Precision
precision= (tp/(tp+fp))*100
print("Precision:",precision)
#Recall
recall= (tp/(tp+fn))*100
print("Recall: ",recall)
#F1 Score
print("F1 score",(2*precision*recall)/(precision + recall))
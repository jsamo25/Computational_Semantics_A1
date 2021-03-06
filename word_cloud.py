import pandas as pd
import nltk
import matplotlib.pyplot as plt

from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS


"""*********************************************************
                    Loading Data
*********************************************************"""

data = pd.read_csv("data/sentiment.csv")


def preprocess(text):
    text = text.str.replace("(<br/>)", "")
    text = text.str.replace("(<a).*(>).*(</a>)", "")
    text = text.str.replace("(&amp)", "")
    text = text.str.replace("(&gt)", "")
    text = text.str.replace("(&lt)", "")
    text = text.str.replace("(\xa0)", " ")
    text = text.str.replace("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "")
    text = text.str.replace("br", "")
    return text


data["text"] = preprocess(data["text"])


"""*********************************************************
                         WordCloud
*********************************************************"""
wordcloud = WordCloud(
    width=1500,
    height=1000,
    background_color="black",
    stopwords=STOPWORDS
).generate(str(data["text"].values))
fig = plt.figure(figsize=(15, 10), facecolor="k", edgecolor="k")
plt.imshow(wordcloud, interpolation="gaussian")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


"""*********************************************************
              Polarity & Subjectivity Tests
*********************************************************"""


def textblob(text):
    polarity, subjectivity = 0, 0
    for text in text:
        analysis = TextBlob(text)
        polarity += analysis.sentiment[0]
        subjectivity += analysis.sentiment[1]
    return [polarity, subjectivity]


data["tokens"] = data["text"].apply(nltk.tokenize.word_tokenize)
data["analysis"] = data["tokens"].apply(textblob)
print(data["analysis"])
data["polarity"] = data["analysis"].apply(lambda x: x[0])
data["subjectivity"] = data["analysis"].apply(lambda x: x[1])

data_train, data_test = train_test_split(data, test_size=0.3)
y_train, y_test = data_train["sentiment"], data_test["sentiment"]
x_train, x_test = (
    data_train[["polarity", "subjectivity"]],
    data_test[["polarity", "subjectivity"]],
)


"""*********************************************************
                    LR & Evaluation
*********************************************************"""
model = LogisticRegression(max_iter=1000, C=0.01).fit(x_train, y_train)
y_pred = model.predict(x_test)


def accuracy(model, x_train, y_train, x_test, y_test):
    print("training set:", model.score(x_train, y_train))
    print("testing set:", model.score(x_test, y_test))


print("\nInitial model score")
accuracy(model, x_train, y_train, x_test, y_test)

# Conclusion: Not relevant to the analysis, better choose other features.

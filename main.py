import pandas as pd

pd.set_option("display.max_columns", 99)
data = pd.read_csv("sentiment10.csv")

#10, 11, 12, 13
print(data.shape)
print(data[:3])
print(data["sentiment"])
print(data["sentiment"].unique())

#15, 16, 17, 18

print(data["sentiment"]=="pos") #checks if each row matches the "pos" value

#assign a boolean value to a pandas column
data["sentiment"] = data["sentiment"] == "pos"
print(data)

data["n_characters"] = data["text"].apply(len)
data["tookens"] = data["text"].apply(...)
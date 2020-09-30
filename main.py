import pandas as pd

pd.set_option("display.max_columns", 99)

#10, 11, 12, 13
data = pd.read_csv("sentiment10.csv")
print(data.shape)
print(data[:3])
print(data["sentiment"])
print(data["sentiment"].unique())


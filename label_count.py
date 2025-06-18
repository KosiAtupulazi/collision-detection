import pandas as pd

df = pd.read_csv("final_dataset/train.csv")
df1 = pd.read_csv("final_dataset/test.csv")
df2 = pd.read_csv("final_dataset/val.csv")

print(df["label"].value_counts())
print(df1["label"].value_counts())
print(df2["label"].value_counts())
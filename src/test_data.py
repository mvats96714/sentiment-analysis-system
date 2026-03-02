import pandas as pd

df = pd.read_csv("data/reviews.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())
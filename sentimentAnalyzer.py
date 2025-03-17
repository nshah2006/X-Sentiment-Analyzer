import pandas as pd

# Load dataset
df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)

# Assign column names
df.columns = ["sentiment", "id", "date", "query", "username", "tweet"]

# Keep only relevant columns
df = df[["sentiment", "tweet"]]

# Convert sentiment labels (0 -> negative, 4 -> positive)
df["sentiment"] = df["sentiment"].replace({0: "negative", 4: "positive"})

# Display the first few rows
print(df.head())

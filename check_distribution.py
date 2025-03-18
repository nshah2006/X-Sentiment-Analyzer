import pandas as pd

# Load dataset
df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ["sentiment", "id", "date", "query", "username", "tweet"]
df = df[["sentiment", "tweet"]]

# Convert labels
df["sentiment"] = df["sentiment"].replace({0: "Negative", 4: "Positive"})

# Count and display sentiment distribution
print("Sentiment Distribution:")
print(df["sentiment"].value_counts())

# Optional: Visualize distribution
import matplotlib.pyplot as plt
df["sentiment"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

print("Checking sentiment distribution before training:")
print(df["sentiment"].value_counts())


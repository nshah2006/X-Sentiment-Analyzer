import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ["sentiment", "id", "date", "query", "username", "tweet"]
df["sentiment"] = df["sentiment"].replace({0: "Negative", 4: "Positive"})

# Plot sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df["sentiment"])
plt.title("Sentiment Distribution in Twitter Dataset")
plt.show()

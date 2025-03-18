import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ["sentiment", "id", "date", "query", "username", "tweet"]
df = df[["sentiment", "tweet"]]

# Convert labels
df["sentiment"] = df["sentiment"].replace({0: "Negative", 4: "Positive"})

# 🔹 Randomly sample 100,000 tweets to get a better representation
df_sampled = df.sample(n=100000, random_state=42)

# ✅ 1️⃣ Sentiment Distribution Pie Chart
plt.figure(figsize=(6, 6))
df_sampled["sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["red", "green"])
plt.title("Sentiment Distribution")
plt.ylabel("")  # Hide y-label
plt.show()

# ✅ 2️⃣ Sentiment Bar Plot
plt.figure(figsize=(6, 4))
sns.countplot(x=df_sampled["sentiment"], palette={"Negative": "red", "Positive": "green"})
plt.title("Sentiment Count Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.show()

# ✅ 3️⃣ WordCloud for Positive & Negative Tweets
positive_tweets = " ".join(df_sampled[df_sampled["sentiment"] == "Positive"]["tweet"])
negative_tweets = " ".join(df_sampled[df_sampled["sentiment"] == "Negative"]["tweet"])

plt.figure(figsize=(12, 6))

# WordCloud for Positive Tweets
plt.subplot(1, 2, 1)
wordcloud = WordCloud(width=500, height=300, background_color="white").generate(positive_tweets)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud for Positive Tweets")

# WordCloud for Negative Tweets
plt.subplot(1, 2, 2)
wordcloud = WordCloud(width=500, height=300, background_color="black").generate(negative_tweets)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud for Negative Tweets")

plt.show()

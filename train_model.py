import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_text
import joblib

# Ensure dataset exists
dataset_path = "dataset/training.1600000.processed.noemoticon.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Error: Dataset file '{dataset_path}' not found! Please check the path.")

# Load dataset
print("Loading dataset...")
df = pd.read_csv(dataset_path, encoding="latin-1", header=None)
df.columns = ["sentiment", "id", "date", "query", "username", "tweet"]
df = df[["sentiment", "tweet"]]

# Convert sentiment labels to readable format
df["sentiment"] = df["sentiment"].replace({0: "negative", 4: "positive"})

# Preprocess tweets
print("Preprocessing tweets...")
df["cleaned_tweet"] = df["tweet"].apply(preprocess_text)

# Convert text to numerical features
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_tweet"])
y = df["sentiment"]

# Free memory by removing unneeded dataframe
del df  

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Ensure 'models/' directory exists
os.makedirs("models", exist_ok=True)

# Save the model & vectorizer
print("\n💾 Saving the trained model...")
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n🎉 Model training complete! The model has been saved in 'models/' directory.")

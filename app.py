import streamlit as st
import joblib
import os
from preprocess import preprocess_text

# Check if model and vectorizer exist
model_path = "models/sentiment_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Error: Model files not found! Please train the model first.")
else:
    # Load trained model
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # UI
    st.title("X Sentiment Analyzer")
    st.write("Enter a tweet below to analyze its sentiment.")

    text = st.text_area("Enter a tweet:")

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Please enter a valid tweet.")
        else:
            cleaned_text = preprocess_text(text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]

            # Display Result
            st.subheader("Sentiment Result")
            if prediction == "positive":
                st.success("✅ Positive Sentiment")
            elif prediction == "negative":
                st.error("❌ Negative Sentiment")
            else:
                st.warning("😐 Neutral Sentiment")


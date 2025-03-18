import streamlit as st
import tweepy
import joblib
import os
import time
import re
import emoji
from preprocess import preprocess_text

# 🎯 Twitter API Credentials (Replace with your keys from developer.twitter.com)
API_KEY = "hZjqhMfiu4Tv4JMd0J2la4occ"
API_SECRET = "I7wpXgXgbzpZEiiAgJMj5ubcNWeR14ZTvplt8Cd9q70GX8GgsZ"
ACCESS_TOKEN = "1901719993894526976-eaU6PKvjtE0gHNUEnJb6XQoy6jCm7N%2B3aeqAjpmJ3IwI%3DnBqERyHGInz9wgMYa8wIfQwZtWvzdQC91feU1VG1joGzAAWQWh"
ACCESS_SECRET = "8xQt6kK3ShlvQyIaki2PFk7hAutKlw5GeFSzhZQXswe6A"

# Authenticate with Tweepy
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# Define model paths
model_path = "models/sentiment_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

# Check if models exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("🚨 Model files not found! Please train the model first.")
    st.stop()

# Load trained model
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# 💡 Streamlit UI Customization
st.set_page_config(page_title="X Sentiment Analyzer", page_icon="💬", layout="wide")

# 🎭 Advanced Emoji Mapping
emoji_map = {
    "happy": ["😃", "😁", "🥳"],
    "love": ["❤️", "😍", "💖"],
    "excited": ["🎉", "🔥", "🚀"],
    "money": ["💰", "💵", "🤑"],
    "angry": ["😡", "🤬", "💢"],
    "sad": ["😢", "😭", "💔"],
    "food": ["🍕", "🍔", "🍣"],
    "sports": ["⚽", "🏀", "🏆"],
    "coding": ["💻", "🐍", "👨‍💻"],
    "technology": ["📱", "🖥️", "🤖"],
    "music": ["🎵", "🎶", "🎧"],
    "weather": ["🌞", "🌧️", "🌈"],
    "travel": ["✈️", "🏝️", "🚗"],
    "animals": ["🐶", "🐱", "🦁"],
}

# Function to fetch tweet text from URL
def get_tweet_text(tweet_url):
    match = re.search(r'/status/(\d+)', tweet_url)
    if not match:
        return None
    tweet_id = match.group(1)

    try:
        tweet = api.get_status(tweet_id, tweet_mode="extended")
        return tweet.full_text
    except Exception as e:
        st.error(f"⚠️ Error fetching tweet: {e}")
        return None

# Function to assign emojis based on words
def get_emojis_for_tweet(tweet_text):
    words = tweet_text.lower().split()  # Tokenize words
    matched_emojis = []

    for word in words:
        if word in emoji_map:
            matched_emojis.extend(emoji_map[word])

    if len(matched_emojis) >= 3:
        return matched_emojis[:3]
    elif len(matched_emojis) > 0:
        return matched_emojis
    else:
        return ["🔍", "💬", "🤔"]

# 🎨 UI Layout
st.markdown("<h1 style='text-align: center;'>🔍 X Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Analyze tweets and detect sentiment in real-time! 💬</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

# ✍🏼 User Input
with col1:
    st.subheader("📝 Enter a Tweet or Twitter URL:")
    tweet_input = st.text_area("Paste a tweet or Twitter link below 👇", height=100, placeholder="E.g. 'I love AI! It's amazing! 🚀' or 'https://twitter.com/user/status/1234567890'")

# 🎨 Analyze Button
if st.button("🔍 Analyze Sentiment"):
    if tweet_input.strip() == "":
        st.warning("⚠️ Please enter a valid tweet or URL!")
    else:
        with st.spinner("Fetching tweet and analyzing sentiment... ⏳"):
            time.sleep(1.5)  # Simulate loading time

            # Check if input is a Twitter URL
            if "twitter.com" in tweet_input or "x.com" in tweet_input:
                tweet_text = get_tweet_text(tweet_input)
                if tweet_text:
                    st.info(f"📢 **Fetched Tweet:**\n> {tweet_text}")
                else:
                    st.error("⚠️ Could not fetch tweet! Please check the link.")
                    st.stop()
            else:
                tweet_text = tweet_input

            # Preprocess and analyze sentiment
            cleaned_text = preprocess_text(tweet_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]

        # 🎯 Display Sentiment Results
        with col2:
            st.subheader("📊 Sentiment Analysis Result")
            if prediction == "positive":
                st.success("✅ **Positive Sentiment!** 😃🎉")
                st.balloons()
            elif prediction == "negative":
                st.error("❌ **Negative Sentiment!** 😞💔")
            else:
                st.warning("😐 **Neutral Sentiment!** 🤔")

        # 🎭 Emoji-Based Sentiment Feedback (Enhanced)
        tweet_emojis = get_emojis_for_tweet(tweet_text)
        sentiment_emoji = {
            "positive": "😃💖🔥",
            "negative": "😡💔🤬",
            "neutral": "😐🤔😶"
        }
        full_emoji_output = f"{sentiment_emoji[prediction]} {' '.join(tweet_emojis)}"
        
        st.subheader("🧐 Emoji-Based Sentiment:")
        st.markdown(f"<h1 style='text-align: center;'>{full_emoji_output}</h1>", unsafe_allow_html=True)

# 🎨 Footer
st.markdown("""
    <hr>
    <p style="text-align: center;">🔹 Built with ❤️ using Streamlit 🔹</p>
""", unsafe_allow_html=True)

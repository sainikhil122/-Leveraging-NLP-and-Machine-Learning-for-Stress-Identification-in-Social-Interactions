import streamlit as st
from models.roberta_sentiment.sentiment_predict import predict_sentiment
from models.roberta_spam.spam_predict import predict_spam
from models.t5_emoji_sentiment.emoji_predict import predict_emoji_sentiment
from models.emotion_model.emotion_predict import predict_emotion
from models.roberta_sarcasm.sarcasm_predict import predict_sarcasm

st.set_page_config(page_title="Stress Detection System", layout="centered")

st.title("Stress Detection")

with st.container():
    st.markdown("### 📝 Input Text")
    text = st.text_area("Enter a message to analyze:")

if st.button("🚀 Analyze"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Analyzing..."):
            sentiment = predict_sentiment(text)
            spam = predict_spam(text)
            emoji_sentiment = predict_emoji_sentiment(text)
            emotion = predict_emotion(text)
            sarcasm = predict_sarcasm(text)

        st.success("✅ Analysis Complete")

        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🗣️ Sentiment:** <span style='color:limegreen'>{sentiment}</span>", unsafe_allow_html=True)
            st.markdown(f"**📮 Spam Detection:** <span style='color:limegreen'>{spam}</span>", unsafe_allow_html=True)
            st.markdown(f"**😂 Emoji Sentiment:** <span style='color:limegreen'>{emoji_sentiment}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"**😢 Emotion:** <span style='color:limegreen'>{emotion}</span>", unsafe_allow_html=True)
            st.markdown(f"**🌀 Sarcasm:** <span style='color:limegreen'>{sarcasm}</span>", unsafe_allow_html=True)

        
        stress_emotions = [
            "anger", "fear", "sadness", "nervousness", "grief",
            "remorse", "disappointment", "annoyance", "disgust"
        ]
        stress_detected = False

        try:
            if sentiment.lower() in ["negative", "very negative"]:
                stress_detected = True
            elif emotion.lower() in stress_emotions:
                stress_detected = True
            elif sarcasm.lower() == "sarcastic" and sentiment.lower() != "positive":
                stress_detected = True
        except Exception:
            st.warning("⚠️ Could not determine stress due to model output format.")

        st.markdown("---")
        st.markdown("### 🧩 Stress Detection Result")
        if stress_detected:
            st.error("🚨 Stress Detected")
        else:
            st.success("✅ No Stress Detected")

st.markdown("---")

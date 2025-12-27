import os
from models.roberta_sentiment.sentiment_model import train_sentiment_model
from models.roberta_spam.spam_model import train_spam_model
from models.roberta_sarcasm.sarcasm_model import train_sarcasm_model
from models.emotion_model.emotion_model import train_emotion_model
from models.t5_emoji_sentiment.emoji_model import train_emoji_model

print("Starting training of all models...")

print("\nTraining Sentiment Model:")
train_sentiment_model(
    data_path="data/Sentiment140.csv",
    model_save_path="models/roberta_sentiment/sentiment_model.pt",
    encoder_path="models/roberta_sentiment/label_encoder.pkl"
)

print("\nTraining Spam Model:")
train_spam_model(
    data_path="data/spam.csv",
    model_save_path="models/roberta_spam/spam_model.pt",
    encoder_path="models/roberta_spam/label_encoder.pkl"
)

print("\nTraining Sarcasm Model:")
train_sarcasm_model(
    data_path="data/train-balanced-sarcasm.csv",
    save_path="models/roberta_sarcasm/sarcasm_model.pt"
)

print("\nTraining Emotion Model:")
train_emotion_model(
    data_path="data/emotion_words.csv",
    model_save_path="models/emotion_model/emotion_model.pt"
)

print("\n Training Emoji Sentiment Model:")
train_emoji_model(
    data_path="data/Emoji_Sentiment_Data_v1.0.csv",
    model_save_path="models/t5_emoji_sentiment/emoji_model.pt",
    encoder_path="models/t5_emoji_sentiment/label_encoder.pkl"
)

print("\nAll models trained successfully!")


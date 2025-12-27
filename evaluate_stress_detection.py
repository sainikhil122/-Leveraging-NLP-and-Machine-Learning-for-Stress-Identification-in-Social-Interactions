import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.predict_stress import predict_stress


df = pd.read_csv("data/test_stress_data1.csv")

actual_labels = []
predicted_labels = []


for i, row in df.iterrows():
    text = row["text"]
    actual = row["label"]
    predicted = predict_stress(text)

    actual_labels.append(actual.lower())
    predicted_labels.append(predicted.lower())

    print(f"\nText: {text}")
    print(f"Predicted: {predicted} | Actual: {actual}")


print("\n--- Evaluation Metrics ---")
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(actual_labels, predicted_labels, target_names=["not stress", "stress"]))

cm = confusion_matrix(actual_labels, predicted_labels, labels=["not stress", "stress"])
labels = ["Not Stress", "Stress"]

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Stress Detection")
plt.tight_layout()
plt.show()




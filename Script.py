# =============================================
# 📘 FAKE NEWS DETECTION SYSTEM
# =============================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional: for text cleaning
import re
import string

# Step 2: Load Data
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Optional: load cleaned dataset if you already have preprocessed data
try:
    cleaned_df = pd.read_csv("cleaned.csv")
    print("Loaded cleaned.csv successfully.")
except FileNotFoundError:
    cleaned_df = None
    print("cleaned.csv not found — proceeding with fake.csv and true.csv only.")

# Step 3: Label Data
fake_df["label"] = 0  # Fake = 0
true_df["label"] = 1  # True = 1

# Step 4: Combine Data
df = pd.concat([fake_df, true_df], axis=0).sample(frac=1).reset_index(drop=True)
print("Dataset shape:", df.shape)

# Step 5: Basic Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

if "text" in df.columns:
    df["text"] = df["text"].apply(clean_text)
else:
    # Try to guess the text column
    text_col = df.columns[df.dtypes == 'object'][0]
    df.rename(columns={text_col: "text"}, inplace=True)
    df["text"] = df["text"].apply(clean_text)

# Step 6: Split Data
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 8: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 9: Predictions
y_pred = model.predict(X_test_tfidf)

# Step 10: Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 11: Save Model and Vectorizer
import joblib
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

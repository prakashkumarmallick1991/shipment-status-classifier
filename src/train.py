import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from preprocess import load_dataset, preprocess_dataframe


# -----------------------------
# Load Dataset
# -----------------------------

df = load_dataset("../data/dataset.json")

df = preprocess_dataframe(df)

X = df["cleaned_text"]
y = df["internalStatus"]


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)


# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test_vec)


# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:\n")
print(accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "../app/model.pkl")

joblib.dump(vectorizer, "../app/vectorizer.pkl")

print("\nModel Saved Successfully")
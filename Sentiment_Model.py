import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- STEP 1: Load data ----
df = pd.read_csv("IMDB_Dataset.csv")
df = df.sample(20000, random_state=42)  

# ---- STEP 2: Clean text FIRST, before anything else touches it ----
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower()

df['review'] = df['review'].apply(clean_text)

# ---- STEP 3: Labels ----
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ---- STEP 4: NOW split (on already-cleaned text) ----
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

# ---- STEP 5: One vectorizer only ----
negations = {'not', 'no', 'nor', 'never', "don't", "isn't", "wasn't", "aren't", "didn't"}
custom_stop_words = list(ENGLISH_STOP_WORDS - negations)

vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=15000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---- STEP 5: Train the model ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---- STEP 6: Test the model ----
y_pred = model.predict(X_test_vec)

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---- STEP 7: Predict on your own sentence ----
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Positive" if prediction == 1 else "Negative"

print("\n--- CUSTOM PREDICTIONS ---")
test_sentences = [
    "This movie was absolutely wonderful, I loved every scene.",
    "Terrible film, complete waste of time.",
]
for sentence in test_sentences:
    print(f"'{sentence}' --> {predict_sentiment(sentence)}")
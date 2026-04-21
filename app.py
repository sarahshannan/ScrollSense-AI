import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------
# Title
# -----------------------
st.title("🧠 ScrollSense AI")
st.subheader("Brain Rot vs Educational Content Classifier")


# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("AI_Term_Project_Dataset.csv")

# Keep only the columns we need
df = df[['text', 'label']]

# Remove rows where text or label is missing
df = df.dropna(subset=['text', 'label'])

# Force labels to numeric
df['label'] = pd.to_numeric(df['label'], errors='coerce')

# Remove rows that still have bad labels after conversion
df = df.dropna(subset=['label'])

# Convert labels to integers
df['label'] = df['label'].astype(int)

# -----------------------
# Clean Text Function
# -----------------------
def clean_text(text):
    if pd.isna(text):   # handles NaN
        return ""
    
    text = str(text)    # convert everything to string
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# -----------------------
# TF-IDF
# -----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# -----------------------
# Train Models
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb_model = MultinomialNB().fit(X_train, y_train)
log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
svm_model = LinearSVC().fit(X_train, y_train)

# -----------------------
# Evaluate Models
# -----------------------
nb_pred = nb_model.predict(X_test)
log_pred = log_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

results = pd.DataFrame({
    "Model": ["Naive Bayes", "Logistic Regression", "SVM"],
    "Accuracy": [
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, log_pred),
        accuracy_score(y_test, svm_pred)
    ],
    "Precision": [
        precision_score(y_test, nb_pred, zero_division=0),
        precision_score(y_test, log_pred, zero_division=0),
        precision_score(y_test, svm_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, nb_pred, zero_division=0),
        recall_score(y_test, log_pred, zero_division=0),
        recall_score(y_test, svm_pred, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, nb_pred, zero_division=0),
        f1_score(y_test, log_pred, zero_division=0),
        f1_score(y_test, svm_pred, zero_division=0)
    ]
})

results_rounded = results.copy()
for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    results_rounded[col] = results_rounded[col].round(3)

# -----------------------
# USER INPUT
# -----------------------

st.markdown("**Try examples like:**")
st.caption("• how to improve memory for exams")
st.caption("• bro this is actually insane")
st.caption("• this is weird but kinda helpful")
user_input = st.text_area("Enter a caption:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)

    if not cleaned.strip():
        st.warning("Please enter a caption with actual words.")
        st.stop()

    vector = vectorizer.transform([cleaned])

    if vector.nnz == 0:
        st.warning("⚠️ I couldn't match this caption to meaningful words from the training dataset.")
        st.info("Try something that sounds like a social media caption, explanation, or reaction.")
        st.stop()

    decision = svm_model.decision_function(vector)[0]
    pred = svm_model.predict(vector)[0]

    st.subheader("📌 Result")

    if abs(decision) < 0.15:
        st.warning("⚠️ Mixed / Ambiguous Content")
        st.write("This caption contains signals from both educational and entertainment language.")
    elif pred == 0:
        st.success("🧠 Educational Content")
    else:
        st.error("⚡ Entertainment / High-Stimulation Content")

    st.write(f"Decision score: {round(decision, 3)}")

    nb_input_pred = nb_model.predict(vector)[0]
    log_input_pred = log_model.predict(vector)[0]
    svm_input_pred = svm_model.predict(vector)[0]

    comparison_df = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "SVM"],
        "Prediction": [
            "Educational" if nb_input_pred == 0 else "Entertainment",
            "Educational" if log_input_pred == 0 else "Entertainment",
            "Educational" if svm_input_pred == 0 else "Entertainment"
        ]
    })

    st.subheader("🤖 Model Predictions for This Caption")
    st.dataframe(comparison_df, use_container_width=True)

    # -----------------------
    # Show Explanation
    # -----------------------
    st.subheader("🔍 Why?")

    feature_names = vectorizer.get_feature_names_out()
    coefficients = log_model.coef_[0]
    word_scores = dict(zip(feature_names, coefficients))

    words = cleaned.split()
    scores = [(w, word_scores.get(w, 0)) for w in words]
    scores_sorted = sorted(scores, key=lambda x: abs(x[1]), reverse=True)

    if scores_sorted:
        for word, score in scores_sorted[:5]:
            direction = "Educational" if score < 0 else "Entertainment"
            st.write(f"**{word}**: {round(score, 3)} ({direction})")
    else:
        st.write("No strong keyword signals found.")

st.divider()

st.subheader("⚙️ Model Comparison")
best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
st.success(f"Best overall model on this test split: {best_model}")

st.dataframe(results_rounded, use_container_width=True)

st.subheader("📈 Accuracy Comparison")
st.bar_chart(results.set_index("Model")[["Accuracy"]])

st.subheader("📈 Precision / Recall / F1 Comparison")
st.bar_chart(results.set_index("Model")[["Precision", "Recall", "F1 Score"]])

st.divider()

# -----------------------
# Dataset Info
# -----------------------
st.subheader("📊 Dataset Info")
st.write("Total samples:", df.shape[0])
st.write("Columns:", df.columns.tolist())
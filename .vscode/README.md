# 🧠 ScrollSense AI

A machine learning dashboard that classifies short-form content captions as:

- Educational (🧠)
- Entertainment / High-Stimulation (⚡)
- Ambiguous (⚠️)

## 🚀 Features

- NLP text classification using TF-IDF
- Model comparison (Naive Bayes, Logistic Regression, SVM)
- Real-time prediction dashboard (Streamlit)
- Explanation of predictions using feature importance
- Handles ambiguous and out-of-scope inputs

## 📊 Models Used

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## 🛠️ How to Run

```bash
pip install streamlit pandas scikit-learn
streamlit run app.py


📁 Dataset

Custom dataset of labeled captions:

0 → Educational
1 → High-Stimulation Entertainment
⚠️ Limitations
Model relies only on text (cannot analyze video content)
Can be fooled by misleading captions
Performance depends on dataset quality


👩‍💻 Author
Sara Shannan
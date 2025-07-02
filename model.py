import numpy as np
import pandas as pd
import string
import pickle
import os
import re
import warnings
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import openpyxl

warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used")
os.makedirs('saved_model', exist_ok=True)

# ✅ Load and sample dataset
dataset = pd.read_csv('IMDB Dataset.csv').sample(2000, random_state=42)
print(f'Dataset loaded: {dataset.shape}')

stopwords = set(pd.read_csv('https://raw.githubusercontent.com/Alir3z4/stop-words/master/english.txt', header=None)[0])
punctuations = string.punctuation

# ✅ Fast regex tokenizer
def fast_tokenizer(text):
    tokens = re.findall(r'\b\w\w+\b', text.lower())
    return [word for word in tokens if word not in stopwords and word not in punctuations]

# ✅ Text cleaner
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [text.strip().lower() for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# ✅ Vectorizer
vectorizer = TfidfVectorizer(tokenizer=fast_tokenizer)

# ✅ Data split
X = dataset['review']
y = dataset['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# ✅ Train + evaluate
def train_model(name, classifier, model_path):
    model = Pipeline([
        ("cleaner", predictors()),
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds, output_dict=True)

    print(f"\n--- {name} ---")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, preds))

    pickle.dump(model, open(model_path, 'wb'))

    return {
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Confusion Matrix": str(cm),
        "Precision_Pos": round(cr["positive"]["precision"], 2),
        "Recall_Pos": round(cr["positive"]["recall"], 2),
        "F1_Pos": round(cr["positive"]["f1-score"], 2),
        "Precision_Neg": round(cr["negative"]["precision"], 2),
        "Recall_Neg": round(cr["negative"]["recall"], 2),
        "F1_Neg": round(cr["negative"]["f1-score"], 2),
    }

# ✅ Train and evaluate
results = []
LRmodel = train_model("Logistic Regression", LogisticRegression(), "saved_model/LinearRegression_model.sav")
results.append(LRmodel)

RFmodel = train_model("Random Forest", RandomForestClassifier(n_estimators=100), "saved_model/RandomForest_model.sav")
results.append(RFmodel)

SVCmodel = train_model("Linear SVC", LinearSVC(), "saved_model/LinearSVC_model.sav")
results.append(SVCmodel)

# ✅ Export to Excel
df = pd.DataFrame(results)
df.to_excel("evaluation_results.xlsx", index=False)
print("\nEvaluation results exported to evaluation_results.xlsx")

# ✅ Load trained models for GUI
LRmodel = pickle.load(open("saved_model/LinearRegression_model.sav", "rb"))
RFmodel = pickle.load(open("saved_model/RandomForest_model.sav", "rb"))
SVCmodel = pickle.load(open("saved_model/LinearSVC_model.sav", "rb"))

# ✅ GUI function
def predict_sentiment():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter a review.")
        return

    lr = LRmodel.predict([text])[0]
    rf = RFmodel.predict([text])[0]
    svc = SVCmodel.predict([text])[0]

    output = f"Logistic Regression: {lr}\nRandom Forest: {rf}\nLinear SVC: {svc}"
    result_label.config(text=output)

# ✅ Build GUI
root = tk.Tk()
root.title("IMDB Review Sentiment Classifier")
root.geometry("650x420")
root.config(bg="#f7f7f7")

tk.Label(root, text="Enter IMDB Review:", font=("Arial", 16), bg="#f7f7f7").pack(pady=10)
entry = tk.Text(root, height=7, width=70, font=("Arial", 12))
entry.pack(padx=10)

tk.Button(root, text="Predict Sentiment", command=predict_sentiment,
          font=("Arial", 14), bg="#4CAF50", fg="white").pack(pady=12)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f7f7f7", fg="black")
result_label.pack(pady=10)

root.mainloop()

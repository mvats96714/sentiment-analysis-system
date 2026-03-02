import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import clean_text


df = pd.read_csv("data/reviews.csv")

df["review"] = df["review"].apply(clean_text)

X = df["review"]
y = df["sentiment"]

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),   
    min_df=2
)
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200,
    C=2,
    solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
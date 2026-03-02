import pickle
from preprocess import clean_text

# Load saved model
model = pickle.load(open("model/model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]


if __name__ == "__main__":
    review = input("Enter your review: ")
    result = predict_sentiment(review)
    print("Predicted Sentiment:", result)
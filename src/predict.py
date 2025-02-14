import joblib

# Load saved model and vectorizer
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Function to predict spam SMS
def predict_spam(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "Spam" if prediction == "spam" else "Not Spam"

# Test prediction
text = input("Enter SMS text: ")
print("Prediction:", predict_spam(text))

from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

# Citation: Using help from Copilot

application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

def load_model():
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)
    return loaded_model, vectorizer

@application.route("/predict", methods=["POST"])
def predict():
    model, vectorizer = load_model()
    data = request.get_json()
    if 'text' not in data:
        return json.dumps({'error': 'No text provided!'}), 400
    text = data['text']
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    result = 'REAL' if prediction == 'REAL' else 'FAKE'
    return json.dumps({'prediction': result})

if __name__ == "__main__":
    application.run(port=5000, debug=True)

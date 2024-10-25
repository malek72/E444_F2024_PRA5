from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

# Load model function
def load_model():
    # Load the trained classifier
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    # Load the vectorizer
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    return loaded_model, vectorizer

# Route for making predictions
@application.route("/predict", methods=["POST"])
def predict():
    # Load the model and vectorizer
    model, vectorizer = load_model()
    
    # Get the input data from the POST request
    data = request.get_json()
    if 'text' not in data:
        return json.dumps({'error': 'No text provided!'}), 400
    
    # Transform the text input using the loaded vectorizer
    text = data['text']
    transformed_text = vectorizer.transform([text])
    
    # Predict using the loaded model
    prediction = model.predict(transformed_text)[0]
    
    # Return the result in JSON format
    result = 'REAL' if prediction == 'REAL' else 'FAKE'
    return json.dumps({'prediction': result})

if __name__ == "__main__":
    application.run(port=5000, debug=True)

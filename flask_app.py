from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

API_KEYS = {
    'developer1': 'passone',
    'developer2': 'passtwo'
}

# Load the pickle file
with open(r'C:\Users\DELL\Desktop\Software\spamfilter\SMS-SPAM-FILTER\Pickle\model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the pickle file
with open(r'C:\Users\DELL\Desktop\Software\spamfilter\SMS-SPAM-FILTER\Pickle\CountVectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

def preprocess_data(df):
    wordnet_lem = WordNetLemmatizer()

    reg_vars = ['http\S+', 'www\S+', 'https\S+', '\W\s+', '\d+', '\t+', '\d+', '\-+', '\\+', '\/+', '\"+', '\#+', '\++', '\@+', '\$+',  '\%+', '\^+', '\&+', '\*+', '\(+', '\)+', '\[+', '\]+', '\{+', '\}+', '\|+', '\;+', '\:+', '\<+', '\>+', '\?+', '\,+', '\.+', '\=+',     '\_+', '\~+', '\`+', '\s+']
    df['text'].replace(reg_vars, ' ', regex=True, inplace=True)
    df['text'] = df['text'].astype(str).str.lower()
    df = df[df['text'].map(lambda x: x.isascii())]
    df['text'] = df.apply(lambda column: nltk.word_tokenize(column['text']), axis=1)
    stopwords = nltk.corpus.stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stopwords])
    df['text'] = df['text'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    df['text'] = df['text'].apply(wordnet_lem.lemmatize)

    processed_data = cv.transform(df['text']).toarray()

    return processed_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized access. Please provide a valid API key.'}), 401


@app.route('/predict', methods=['POST'])
def predict():
    api_key = request.headers.get('API-Key')
    logging.info(f'Received request with API key: {api_key}')
    
    if api_key is None:
        logging.warning('API key is missing in the request headers.')
        return jsonify({'error': 'API key is missing'}), 401
    
    if api_key not in API_KEYS.values():
        logging.warning('Invalid API key provided.')
        return jsonify({'error': 'Invalid API key'}), 401

    # Get the input data from the JSON payload
    text = request.json.get('text')

    if text is None:
        logging.warning('Text data is missing in the request JSON payload.')
        return jsonify({'error': 'Text data is missing'}), 400

    # Preprocess the data
    processed_data = preprocess_data(pd.DataFrame({'text': [text]}))

    # Use the model to make a prediction
    prediction = model.predict(processed_data)

    # Return prediction result as JSON
    result = 'Not Spam' if prediction[0] == 0 else 'Spam'
    return jsonify({'prediction': result})


if __name__ == "__main__":
    app.run(debug=True)

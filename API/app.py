
<<<<<<< Updated upstream
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
=======
from flask import Flask, jsonify, render_template, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
import os
>>>>>>> Stashed changes

import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Deep Learning'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Deep Learning'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,
                  config=swagger_config)


<<<<<<< Updated upstream
max_features = 100000
=======

# Tokenizing (?)
max_features = 1000
>>>>>>> Stashed changes
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

def cleansing(sent):

    string = sent.lower()

    string = re.sub(r'[^a-zA-z0-9]', ' ', string)
    return string

#rnn
file = open("",'rb')
feature_file_from_rnn = pickle.load(file)
file.close()

model_file_from_rnn = load_model('')

#lstm
file = open("",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('')

<<<<<<< Updated upstream
#rnn 
@swag_from("", methods=['POST'])
@app.route('', methods=['POST'])
def rnn():
=======
    model_file_from_nn = pickle.load(open("API/model_of_nn/model.p", 'rb'))
    file.close()
    # <<End of Loading NN Model>>
>>>>>>> Stashed changes

    original_text = request.form.get('text')
    original_text = ['original_text']
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(original_text)
    
    text = tfidf_vect.transform(original_text)

<<<<<<< Updated upstream
    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

    prediction = model_file_from_rnn.predict(feature)
=======
    prediction = model_file_from_nn.predict(text)
>>>>>>> Stashed changes
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using RNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#lstm
@swag_from("", methods=['POST'])
@app.route('', methods=['POST'])
def lstm():

    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]


    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()



















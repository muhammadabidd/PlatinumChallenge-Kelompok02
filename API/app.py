
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



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


max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

def cleansing(sent):

    string = sent.lower()

    string = re.sub(r'[^a-zA-z0-9]', '', string)
    return string

#NN
file = open("",'rb')
feature_file_from_rnn = pickle.load(file)
file.close()

model_file_from_rnn = load_model('')

#LSTM
file = open("",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('')

#Connecting NN
@swag_from("", methods=['POST'])
@app.route('', methods=['POST'])
def rnn():

    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

    prediction = model_file_from_rnn.predict(feature)
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

#Connecting LSTM
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



















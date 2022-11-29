
from flask import Flask, jsonify, render_template, request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import os

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



# Tokenizing (?)
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']


# Cleansing (?)
def cleansing(sent):

    string = sent.lower()

    string = re.sub(r'[^a-zA-z0-9]', '', string)
    return string



#Text NN
@swag_from("docs/Text.yml", methods=['POST'])
@app.route('/neural_network_text', methods=['POST'])
def neural_network_text():

    # <<Start of Making NN Model>>
    file = open("model_of_nn/model.p",'rb')
    feature_file_from_nn = pickle.load(file)
    file.close()

    model_file_from_nn = load_model('')
    # <<End of Loading NN Model>>

    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_nn.shape[1])

    prediction = model_file_from_nn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]


    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using NN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#Text LSTM
@swag_from("docs/Text.yml", methods=['POST'])
@app.route('/lstm_text', methods=['POST'])
def lstm_text():

    #Start of Loading LSTM Model
    file = open("",'rb')
    feature_file_from_lstm = pickle.load(file)
    file.close()

    model_file_from_lstm = load_model('')
    #End of Loading LSTM Model

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

#File Neural Network
@swag_from("docs/file_Upload.yml", methods = ['POST'])
@app.route("/neural_network_file", methods=["POST"])
def neural_network_file():
    file = request.files['file'] #the Function is not designet yet

    
    json_response = {
        'status_code' : 200,
        'description' : "File yang sudah diproses",
        'data' : "Its Functioned",
    }

    response_data = jsonify(json_response)
    return response_data

#File Neural Network
@swag_from("docs/file_Upload.yml", methods = ['POST'])
@app.route("/lstm_file", methods=["POST"])
def lstm_file():
    file = request.files['file'] #the Function is not designet yet

    
    json_response = {
        'status_code' : 200,
        'description' : "File yang sudah diproses",
        'data' : "Its Functioned",
    }

    response_data = jsonify(json_response)
    return response_data





if __name__ == '__main__':
    app.run(debug=True)



















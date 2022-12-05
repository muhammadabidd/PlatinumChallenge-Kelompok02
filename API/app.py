

import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from NeuralNetwork import get_centiment_nn

import json
import pandas as pd
import numpy as np
import os
import re
from werkzeug.utils import secure_filename
from datetime import datetime
from Data_Cleansing import process_text
from Data_Cleansing import preprocess
from flask import Flask, jsonify, make_response, render_template, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
# from flasgger import make_response



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


#Text NN
@swag_from("docs/Text.yml", methods=['POST'])
@app.route('/neural_network_text', methods=['POST'])
def neural_network_text():

    #<<Start of Loading Neural Network Model and Feature>>
    #Import model
    file = open("API/resources_of_nn_countvectorizer/model_countvectorizer_nn.pickle",'rb')
    model = pickle.load(file)
    file.close()

    #Import Feature
    file = open("API/resources_of_nn_countvectorizer/feature_countvectorizer_nn.pickle",'rb')
    feature = pickle.load(file)
    file.close()
    #<<End of Loading Neural Network Model and Feature>>

    # Processing text
    def get_centiment_nn(original_text):
        text = feature.transform([process_text(original_text)])
        result = model.predict(text)[0]
        return result

    original_text = request.form.get('text')
    get_sentiment = get_centiment_nn(original_text)
   
    
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

    #<<Start of Loading LSTM Model>>
    file = open("API/resources_of_lstm/x_pad_sequences.pickle",'rb')
    feature_file_from_lstm = pickle.load(file)
    file.close()

    model_file_from_lstm = load_model('API/resources_of_lstm/model.h5')
    #<<End of Loading LSTM Model>>

    #Getting text input
    original_text = request.form.get('text')

    #Cleaning inputted text
    text = [process_text(original_text)]

    #Feature extraction
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    #predicting
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    #Response
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

#Defining allowed extensions
allowed_extensions = set(['csv'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions



#File Neural Network
@swag_from("docs/file_Upload.yml", methods = ['POST'])
@app.route("/neural_network_file", methods = ["POST"])
def neural_network_file():
    file = request.files['file'] 

    if file and allowed_file(file.filename):

        # <<making new file from inputted file>>
        filename = secure_filename(file.filename)
        time_stamp = (datetime.now().strftime('%d-%m-%Y_%H%M%S'))

        new_filename = f'{filename.split(".")[0]}_{time_stamp}.csv'
        
        # <<saving new inputted file>>
        save_location = os.path.join('API/input', new_filename)
        file.save(save_location)
        filepath = 'API/input/' + str(new_filename)


        # <<reading csv file>>
        data = pd.read_csv(filepath, encoding='latin-1')
        print(data)
        first_column_pre_process = data.iloc[:, 1]

        #<<Start of Loading Neural Network Model and Feature>>
        file = open("API/resources_of_nn_countvectorizer/model_countvectorizer_nn.pickle",'rb')
        model = pickle.load(file)
        file.close()

        #Import Feature
        file = open("API/resources_of_nn_countvectorizer/feature_countvectorizer_nn.pickle",'rb')
        feature = pickle.load(file)
        file.close()
        #<<End of Loading Neural Network Model and Feature>>

        # Processing text
        def get_centiment_nn(original_text):
            text = feature.transform([process_text(original_text)]) 
            result = model.predict(text)[0]
            return str(result)

        sentiment_result = []

        for text in first_column_pre_process:
            #Cleaning inputted text
            file_clean = process_text(text)
            print('ini teksnya : ', file_clean)

            get_sentiment = get_centiment_nn(file_clean)
            print('ini sentimennya : ', get_sentiment)

            sentiment_result.append(get_sentiment)
    
            
        new_data_frame = pd.DataFrame(
                {'text': first_column_pre_process,
                'Sentiment': sentiment_result,
                })
                
        outputfilepath = f'API/output/{new_filename}'
        new_data_frame.to_csv(outputfilepath)

        
        result = new_data_frame.to_json(orient="index")
        parsed = json.loads(result)
        json.dumps(parsed) 


    json_response = {
        'status_code' : 200,
        'description' : "File sudah diproses",
        'result' : parsed
    }
        

    response_data = jsonify(json_response)
    return response_data



    


#File LSTM
@swag_from("docs/file_Upload.yml", methods = ['POST'])
@app.route("/lstm_file", methods=["POST"])
def lstm_file():
    file = request.files['file'] 

    if file and allowed_file(file.filename):

        # <<making new file from inputted file>>
        filename = secure_filename(file.filename)
        time_stamp = (datetime.now().strftime('%d-%m-%Y_%H%M%S'))

        new_filename = f'{filename.split(".")[0]}_{time_stamp}.csv'
        
        # <<saving new inputted file>>
        save_location = os.path.join('API/input', new_filename)
        file.save(save_location)
        filepath = 'API/input/' + str(new_filename)


        # <<reading csv file>>
        data = pd.read_csv(filepath, encoding='latin-1')

        first_column_pre_process = data.iloc[:, 1]

        #<<Start of Loading LSTM Model>>
        file = open("API/resources_of_lstm/x_pad_sequences.pickle",'rb')
        feature_file_from_lstm = pickle.load(file)
        file.close()

        model_file_from_lstm = load_model('API/resources_of_lstm/model.h5')
        #<<End of Loading LSTM Model>>

        # <<processing text>>
        sentiment_result = []

        for text in first_column_pre_process:
            #Cleaning inputted text
            file_clean = [process_text(text)]

        
            #Feature extraction
            feature = tokenizer.texts_to_sequences(file_clean)
            feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

            #predicting
            prediction = model_file_from_lstm.predict(feature)
            get_sentiment = sentiment[np.argmax(prediction[0])]

            sentiment_result.append(get_sentiment)


        new_data_frame = pd.DataFrame(
                {'text': first_column_pre_process,
                'Sentiment': sentiment_result,
                })

        outputfilepath = f'API/output/{new_filename}'
        new_data_frame.to_csv(outputfilepath)

        
        result = new_data_frame.to_json(orient="index")
        parsed = json.loads(result)
        json.dumps(parsed) 


    json_response = {
        'status_code' : 200,
        'description' : "File sudah diproses",
        'result' : parsed
    }

    response_data = jsonify(json_response)
    return response_data




# Error Handling
@app.errorhandler(400)
def handle_400_error(_error):
    "Return a http 400 error to client"
    return make_response(jsonify({'error': 'Misunderstood'}), 400)


@app.errorhandler(401)
def handle_401_error(_error):
    "Return a http 401 error to client"
    return make_response(jsonify({'error': 'Unauthorised'}), 401)


@app.errorhandler(404)
def handle_404_error(_error):
    "Return a http 404 error to client"
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(500)
def handle_500_error(_error):
    "Return a http 500 error to client"
    return make_response(jsonify({'error': 'Server error'}), 500)



if __name__ == '__main__':
    app.run(debug=True)



















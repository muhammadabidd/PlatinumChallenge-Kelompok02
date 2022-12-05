import pandas as pd

df = pd.read_csv('./DataTrain/train_preprocess.tsv', delimiter='\t', header=None, )
df.columns=['text', 'label']

df.label.value_counts()

# Making Database

import sqlite3

db = sqlite3.connect('platinum.db')
mycursor = db.cursor()
query = "CREATE TABLE IF NOT EXISTS Table_1 (text varchar(255), label varchar(255));"
mycursor.execute(query)
db.commit()

#Cleansing
import re


def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n','', text)
    text = re.sub('rt','', text)
    text = re.sub('user','', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub('  +',' ', text)
    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub('  +',' ', text) 
    return text


# Untuk Proses Cleaning Data
def preprocess(text):
    text = lowercase(text) # 1
    text = remove_unnecessary_char(text) # 2
    text = remove_nonaplhanumeric(text) # 3
    return text


def process_text(input_text):
    try: 
        output_text = preprocess(input_text)
        return output_text
    except:
        print("Text is unreadable")


df['text_clean'] = df.text.apply(process_text)

# Test
text = '[][{warung ini dimiliki oleh pengusaha pabrik tahu yang sudah puluhan tahun terkenal membuat tahu putih di bandung . tahu berkualitas , dipadu keahlian memasak , dipadu kretivitas , jadilah warung yang menyajikan menu utama berbahan tahu , ditambah menu umum lain seperti ayam . semuanya selera indonesia . harga cukup terjangkau . jangan lewatkan tahu bletoka nya , tidak kalah dengan yang asli dari tegal !'
text = process_text(text)


# Kita simpan teks ke dalam sebuah variabel
data_preprocessed = df.text_clean.tolist()

# Untuk melakukan Feature Extraction, kita menggunakan library "Sklearn atau scikit-learn".
# Sklearn adalah library untuk melakukan task-task Machine Learning.
# "CountVectorizer" merupakan salah satu modul untuk melakukan "BoW"
from sklearn.feature_extraction.text import CountVectorizer

# Kita proses Feature Extraction
count_vect = CountVectorizer()
count_vect.fit(data_preprocessed)

X = count_vect.transform(data_preprocessed)
print ("Feature Extraction selesai")

import pickle

pickle.dump(count_vect, open("feature(countvetorizer_nn).p", "wb"))

from sklearn.model_selection import train_test_split

classes = df.label

X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size = 0.2)

from sklearn.neural_network import MLPClassifier

model = MLPClassifier() 
model.fit(X_train, y_train)

print ("Training selesai")

pickle.dump(model, open("model(countvetorizer_nn).p", "wb"))

from sklearn.metrics import classification_report

test = model.predict(X_test)

print ("Testing selesai")

print(classification_report(y_test, test)) 

# Untuk lebih menyakinkan lagi, kita juga bisa melakukan "Cross Validation"
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5,random_state=42,shuffle=True)

accuracies = []

y = classes

for iteration, data in enumerate(kf.split(X), start=1):

    data_train   = X[data[0]]
    target_train = y[data[0]]

    data_test    = X[data[1]]
    target_test  = y[data[1]]

    clf = MLPClassifier()
    clf.fit(data_train,target_train)

    preds = clf.predict(data_test)

    # for the current fold only    
    accuracy = accuracy_score(target_test,preds)

    print("Training ke-", iteration)
    print(classification_report(target_test,preds))
    print("======================================================")

    accuracies.append(accuracy)

# this is the average accuracy over all folds
average_accuracy = np.mean(accuracies)

print()
print()
print()
print("Rata-rata Accuracy: ", average_accuracy)

def get_centiment_nn(original_text):
    text = count_vect.transform([process_text(original_text)])
    result = model.predict(text)[0]
    return result
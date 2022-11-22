import pandas as pd
import re
import sqlite3

db = sqlite3.connect('challenge.db')
mycursor = db.cursor()
query = "CREATE TABLE IF NOT EXISTS nama_tabel (nama_kolom1 tipe_kolom1, nama_kolom2 tipe_kolom2);"
mycursor.execute(query)
db.commit()

db = sqlite3.connect('challenge.db', check_same_thread=False)
db.text_factory = bytes
mycursor = db.cursor()
q_kamusalay = "select * from kamusalay"
t_kamusalay = pd.read_sql_query(q_kamusalay, db)

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ', text)
    text = re.sub('rt',' ', text)
    text = re.sub('user',' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub('  +',' ', text)
    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub('  +',' ', text) 
    return text

alay_dict_map = dict(zip(t_kamusalay['alay'], t_kamusalay['cleaned'])) #zip menyatukan value dengan index yang sama
def normalize_alay(text):
    for word in alay_dict_map:
        return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

"""
for word in text.split(' '):
    if word in alay_dict_map:
        return(' '.join([alay_dict_map[word])
    else:
        return(' '.join(word)
"""

# Untuk Proses Cleaning Data
def preprocess(text):
    text = lowercase(text) # 1
    text = remove_unnecessary_char(text) # 2
    text = remove_nonaplhanumeric(text) # 3
    text = normalize_alay(text) # 4
    return text


# Untuk Proses File CSV
def process_csv(input_file):
    first_column = input_file.iloc[:, 0]
    print(first_column)

    for tweet in first_column:
        tweet_clean = preprocess(tweet)
        query_tabel = "insert into tweet (tweet_kotor,tweet_bersih) values (?, ?)"
        val = (tweet, tweet_clean)
        mycursor.execute(query_tabel, val)
        db.commit()
        print(tweet)


# Untuk Proses Text
def process_text(input_text):
    try: 
        output_text = preprocess(input_text)
        return output_text
    except:
        print("Text is unreadable")
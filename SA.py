import tensorflow as tf
import csv
from tqdm import tqdm
import re
import numpy as np
from collections import Counter
import os
#import pickle

#####################
#PREPROCESS DATA
#####################
def preprocess_data():
    with open('./data/twitter.csv') as csv_file:
        labels = []
        train = []

        allData = ""
        csv_reader = csv.reader(csv_file, delimiter=',')

        print("Filtering data phase 1...")

        try:
            for row in tqdm(csv_reader):
                if int(row[0]) == 0:
                    labels += [0]
                else:
                    labels += [1]
                filtered = re.sub('[.?!W#@,]', '', row[5])
                filtered = filtered.lower()
                allData += filtered + " "
                train += [filtered.split(" ")]
        except:
            pass

        print("Filtering data phase 2...")
        allWords = allData.split(" ")
        del allData
        counter = Counter(allWords)
        del allWords
        most_occur = counter.most_common(400000)
        del counter

        print("Filtering data phase 3...")
        vocab = [set[0] for set in most_occur]
        word2idx = {word:idx for idx, word in enumerate(vocab)}
        idx2word = {idx:word for idx, word in enumerate(vocab)}

        '''with open('word_Id.p', 'wb') as fp:
            pickle.dump(word2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)'''

        del vocab
        del most_occur

        print("Filtering data phase 4...")
        vectors = []
        for i in tqdm(range(len(train))):
            vectors += [[]]
            for j in train[i]:
                try:
                    vectors[i] += [word2idx[j]]
                except:
                    vectors[i] += [400001]

        del word2idx
        del idx2word
        del train

        print("Filtering data phase 5...")
        data = []
        for i in tqdm(range(len(vectors))):
            data += [(vectors[i], labels[i])]

        del vectors
        del labels

        np.random.shuffle(data)

        print("Filtering data phase 6...")
        vectors = []
        labels = []
        for i in tqdm(data):
            vectors += [i[0]]
            labels += [i[1]]

        del data

        return vectors, np.array(labels)

#####################
#CREATE MODEL
#####################
def build_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

#####################
#FUNCTION CALLS
#####################
train_x, train_y = preprocess_data()

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_x,
    value=400002,
    padding='post',
    maxlen=20)

model = build_model(400003)

model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(train_data,
    train_y,
    epochs=1,
    batch_size=64)

model.save("saved_model/model")

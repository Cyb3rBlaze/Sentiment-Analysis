import tensorflow as tf
import numpy as np
import pickle
import sys
import speech_recognition as sr

#####################
#LOADING INDEXES
#####################
word2idx = None
with open('word_Id.p', 'rb') as fp:
    word2idx = pickle.load(fp)

def vectorize(sentence):
    words = sentence.split(" ")
    vectors = []
    for i in words:
        try:
            vectors += [word2idx[i]]
        except:
            vectors += [400001]

    if len(vectors) > 30:
        return vectors[:30]
    else:
        difference = 30 - len(vectors)
        for i in range(difference):
            vectors += [400002]
        return vectors

#####################
#RUNNING PREDICTIONS
#####################
model = tf.keras.models.load_model('saved_model/model')

r = sr.Recognizer()
mic = sr.Microphone()

print("Listening...")

while True:
    with mic as source:
        audio = r.listen(source)

    text = r.recognize_google(audio)

    if text == "exit":
        sys.exit()
    model_input = vectorize(text)
    prediction = model.predict([model_input])
    print(prediction)
    if prediction[0][0] >= 0.5:
        print("Predicted positive!")
    else:
        print("Predicted negative!")

import pickle, json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import random

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('Bot.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('testbot.h5')


def chatbot(input, confidence=0.75):
    # tokenize
    sentence_words = nltk.word_tokenize(input)
    # lemmatize
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    # create binary pattern
    bag = [0] * len(words)
    for i in sentence_words:
        for index, word in enumerate(words):
            if i == word:
                bag[index] = 1
    bow = np.array(bag)
    # get result from the model
    res = model.predict(np.array([bow]))[0]
    results = []
    # check weather model is confident ( here is > 25)
    for i, r in enumerate(res):
        if r > confidence:
            results.append([i, r])
    results.sort(key=lambda x: x[1], reverse=True)
    # get class from classes
    result_list = []
    for r in results:
        result_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    print(result_list)
    tag = result_list[0]['intent']
    # check tag with main json and choose randon response
    result = ''
    for i in intents['intents']:
        if i['tag'] == tag:
            result = random.choice(i['responses'])

    return result


while True:
    input_data = input('')
    print(chatbot(input_data))



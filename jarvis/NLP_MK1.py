from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle, json, nltk, random
import numpy as np


class chatbot:
    def __init__(self, model_name, confidence_rate=0.75):
        self.model_name = model_name
        self.confidence_rate = confidence_rate
        self.model = load_model(self.model_name)
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('Bot.json').read())
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))

    def print_parameters(self):
        print('Model Complied', self.model_name)
        print('Confidence rate', self.confidence_rate)

    def ask(self, input_data):
        sentence_words = nltk.word_tokenize(input_data)
        # lemmatize
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        # create binary pattern
        bag = [0] * len(self.words)
        for i in sentence_words:
            for index, word in enumerate(self.words):
                if i == word:
                    bag[index] = 1
        bow = np.array(bag)
        # get result from the model
        res = self.model.predict(np.array([bow]))[0]
        results = []
        # check weather model is confident ( here is > 25)
        for i, r in enumerate(res):
            if r > self.confidence_rate:
                results.append([i, r])
        results.sort(key=lambda x: x[1], reverse=True)
        # get class from classes
        result_list = []
        for r in results:
            result_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        print(result_list)
        tag = result_list[0]['intent']
        # check tag with main json and choose randon response
        result = ''
        for i in self.intents['intents']:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                # result = result.format(model='chatbot')
        return result , tag


if __name__ == '__main__':
    jarvis = chatbot('testbot.h5', 0.98)
    jarvis.print_parameters()
    while True:
        try:
            input_data = input('')
            reply , tag = jarvis.ask(input_data)
            print("---",reply , tag)
        except:
            print("Sorry didnt get it can you say once again.")
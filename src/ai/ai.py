# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import pymorphy2

import sys
import numpy as np
from .data import read

class AI():
    def __init__(self):
        #mytext = text
        self.className = ["Аппендицит", "Гастрит", "Гепатит", "Дуоденит", "Колит", "Панкреатит", "Холицестит", "Эзофагит", "Энтерит", "Язва"]  # Объявляем интересующие нас классы
        self.model = read.load_tensor_model()
        self.dictionary_ = read.get_dict()
        self.nClasses = len(self.className)  # Считаем количество классов
        

    def predict(self, text):
        self.mytext = text
        self.morph = pymorphy2.MorphAnalyzer()
        # Преобразовываем текст в последовательность индексов согласно частотному словарю
        self.mytext = list(filter(None, re.split('\W', self.mytext))) # разбиение текста на список слов и цифр
        
        for i in range(len(self.mytext)):
            self.mytext[i] = self.morph.parse(self.mytext[i])[0].normal_form

        # Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
        self.testWordIndexes = self.token(self.mytext)
        self.tokenizer = Tokenizer(num_words=len(self.testWordIndexes), filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff', lower=True, split=' ', oov_token='unknown', char_level=False)

        self.xTest = np.array([self.testWordIndexes])

        self.xTest01 = self.tokenizer.sequences_to_matrix(self.xTest.tolist()) # Подаем xTest в виде списка, чтобы метод успешно сработал
        return self.__match()

    def token(self, text):
        
        self.testWordIndexes = []
        maxWordsCount = 100

        for i in range(len(text)):
            flag = 1
            for j in range(len(self.dictionary_)):
                if (text[i] == self.dictionary_[j][0]) and flag:
                    self.testWordIndexes.append(self.dictionary_[j][1])
                    flag = 0
            if flag:
                self.testWordIndexes.append(1)

        if len(self.testWordIndexes) < maxWordsCount:
            for i in range(maxWordsCount - len(self.testWordIndexes)):
                self.testWordIndexes.append(1)
        elif len(self.testWordIndexes) > maxWordsCount:
            for i in range(len(self.testWordIndexes) - maxWordsCount):
                self.testWordIndexes.pop()

        return self.testWordIndexes

    def __match(self):
        answer=''

        for i in range(self.nClasses):
            totalSumRec = 0
            currPred = self.model.predict(self.xTest01)
            currOut = np.argmax(currPred, axis=1)
            evVal = []

            for j in range(self.nClasses):
                evVal.append(len(currOut[currOut==j])/len(self.xTest01))

            totalSumRec += len(currOut[currOut == i])
            recognizedClass = np.argmax(evVal)  # Определяем, какой класс в итоге за какой был распознан

            # Выводим результаты распознавания по текущему классу
            isRecognized = "Это НЕПРАВИЛЬНЫЙ ответ!"
            if (recognizedClass == i):
                isRecognized = "Это ПРАВИЛЬНЫЙ ответ!"
                answer = self.className[i]

            str1 = 'Класс: ' + self.className[i] + " " * (11 - len(self.className[i])) + str(
                int(100 * evVal[i])) + "% сеть отнесла к классу " + self.className[recognizedClass]
            print(str1, " " * (55 - len(str1)), isRecognized, sep='')


        print('Правильный ответ: ' + answer)
        return answer

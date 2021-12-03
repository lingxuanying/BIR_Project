from django.shortcuts import render
import gensim
from gensim.models import Word2Vec
import os
from django.conf import settings
import matplotlib.pyplot as plt
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
import pandas as pd
import re
import plotly
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')




# Create your views here.
def index(request):
    return render(request, "index.html") #必须用这个return

def search(request, text):
    input_word = text
    ps = PorterStemmer()
    input_word = ps.stem(input_word)
    #print(input_word)
    myframe = pd.read_csv(os.path.join(settings.BASE_DIR, 'pos_tag+tfidf(1).csv'), encoding='utf-8', engine='python')
    target_col = []
    for i in list(myframe.columns):
        if(ps.stem(i.split('%')[0])==input_word):
            target_col.append(i)

    target_abstract = []
    for i in range(len(myframe)):
        find = False
        for j in target_col:
            if(myframe[j][i]>0 and not find):
                find = True
                if(i<50):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\AKI\\'+str(i//10+1)+'.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                elif (i >= 50 and i < 100):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\diabetes mellitus\\' + str(i // 10 + 1 - 5) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                elif (i >= 100 and i < 150):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\heart disease\\' + str(i // 10 + 1 - 10) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                else:
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\lung cancer\\' + str(i // 10 + 1 - 15) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                target_abstract.append([i, data['abstract'][i % 10]])

    target_sentence = []
    for i in target_abstract: # i for every article
        for j in sent_tokenize(i[1]): # j for every sentence
            find = False
            temp = []
            for k in word_tokenize(j): # k for every words
                temp.append(k.lower())
            tokens_tag = pos_tag(temp)
            for k in range(len(tokens_tag)):
                temp[k] = temp[k] + '%' + tokens_tag[k][1].lower()
                for l in target_col:
                    if(l==temp[k] and not find):
                        find = True
                        sentence = j

                        target_sentence.append([round(myframe[l][i[0]], 2), sentence.replace(temp[k].split('%')[0], "<mark>" + temp[k].split('%')[0] + "</mark>")])
    target_sentence = sorted(target_sentence, reverse = True)

    return render(request, "result.html", {'target_sentence': target_sentence})



def statistics(request):
    return render(request, "statistics.html") #必须用这个return

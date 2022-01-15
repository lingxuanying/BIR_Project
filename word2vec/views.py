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
from gensim import corpora
from gensim.summarization import bm25
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

input_word = ""


# Create your views here.
def index(request):
    return render(request, "index.html") #必须用这个return

def cal_bm25(input_word, article_row):
    p_stemmer = PorterStemmer()
    global tokenizer
    text = []
    # print(article_row)
    if (not pd.isnull(article_row)):
        # print(article_row)
        article_row = tokenizer.tokenize(article_row)
        article_list = []

        for a in article_row:
            a_split = a.replace('?', ' ').replace('(', ' ').replace(')', ' ').split(' ')
            # 詞干提取
            stemmed_tokens = [p_stemmer.stem(i) for i in a_split]
            article_list.append(stemmed_tokens)

        #print(article_list)

        query = [input_word]
        query_stemmed = [p_stemmer.stem(i) for i in query]
        print('query_stemmed :', query_stemmed)

        # bm25模型
        bm25Model = bm25.BM25(article_list)
        # 逆文件頻率
        average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
        scores = bm25Model.get_scores(query_stemmed, average_idf)
        print('scores :', scores)
        print(bm25Model.idf)

def search(request, tf_form, text, type):
    global input_word
    ps = PorterStemmer()

    if(text!='none' and input_word!=ps.stem(text)):
        input_word = text
        input_word = ps.stem(input_word)

    tfidf = pd.read_csv(os.path.join(settings.BASE_DIR, 'tfidf (2).csv'), encoding='utf-8', engine='python')
    #print(input_word)
    target_col = []
    for i in list(tfidf.columns):
        if(ps.stem(i.split('%')[0])==input_word):
            target_col.append(i)

    target_abstract = []
    num_per_class = 250
    target_sentence = []
    for i in range(len(tfidf)):
        find = False
        for j in target_col:
            if(tfidf[j][i]>0 and not find):
                find = True
                if(i < num_per_class):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\AKI\\'+str(i//10+1)+'.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                elif (i >= num_per_class and i < 2 * num_per_class):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\diabetes mellitus\\' + str(i // 10 + 1 - 25) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                elif (i >= 2 * num_per_class and i < 3 * num_per_class):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\heart disease\\' + str(i // 10 + 1 - 50) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                else:
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'data\\covid19\\' + str(i // 10 + 1 - 75) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                #cal_sentence(i, data['abstract'][i % 10])
                for k in sent_tokenize(data['abstract'][i % 10]):  # j for every sentence
                    find = False
                    temp = []
                    for l in word_tokenize(k):  # k for every words
                        temp.append(l.lower())
                    tokens_tag = pos_tag(temp)
                    #print(tokens_tag)
                    for l in range(len(tokens_tag)):
                        temp[l] = temp[l] + '%' + tokens_tag[l][1].lower()
                        for m in target_col:
                            if (m == temp[l] and not find):
                                find = True
                                score = tfidf[m][i]

                                if (type == "aki" and i < num_per_class):
                                    score = score * 2
                                elif (type == "diabetes_mellitus" and i >= num_per_class and i < 2 * num_per_class):
                                    score = score * 2
                                elif (type == "heart_disease" and i >= 2 * num_per_class and i < 3 * num_per_class):
                                    score = score * 2
                                elif (type == "covid19" and i >= 3 * num_per_class and i < 4 * num_per_class):
                                    score = score * 2


                                #cal_bm25(temp[l].split('%')[0], k)
                                target_sentence.append([round(score, 2), k.replace(" " + temp[l].split('%')[0], " <mark>" + temp[l].split('%')[0] + "</mark>")])
    target_sentence = sorted(target_sentence, reverse = True)
    return render(request, "result.html", {'target_sentence': target_sentence[:50]})


def statistics(request):
    return render(request, "statistics.html") #必须用这个return

def structure(request):
    return render(request, "structure.html") #必须用这个return

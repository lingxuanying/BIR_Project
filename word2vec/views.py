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
import random
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

input_word = ""


# Create your views here.
def index(request):
    return render(request, "index.html") #必须用这个return

def cal_bm25(input_word, article_row):
    p_stemmer = PorterStemmer()
    for i in range(len(input_word)):
        input_word[i] = input_word[i].split('%')[0]
    #print(input_word)
    global tokenizer
    text = []
    # print(article_row)
    scores = []
    if (len(article_row)>0):
        article_list = []
        for a in article_row:
            a_split = a.replace('?', ' ').replace('(', ' ').replace(')', ' ').split(' ')
            # 詞干提取
            stemmed_tokens = [p_stemmer.stem(i) for i in a_split]
            article_list.append(stemmed_tokens)

        query = input_word
        query_stemmed = [p_stemmer.stem(i) for i in query]

        # bm25模型
        bm25Model = bm25.BM25(article_list)
        # 逆文件頻率
        average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
        scores = bm25Model.get_scores(query_stemmed, average_idf)
    return scores

def search(request, tf_form, text, type):
    global input_word
    ps = PorterStemmer()

    if(text!='none' and input_word!=ps.stem(text)):
        input_word = text
        input_word = ps.stem(input_word)

    tfidf = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\tfidf (2).csv'), encoding='utf-8', engine='python')
    #print(input_word)
    target_col = []
    for i in list(tfidf.columns):
        if(ps.stem(i.split('%')[0])==input_word):
            target_col.append(i)

    num_per_class = 250
    target_sentence = []
    target_sentence_bm25 = []
    type_list = []
    for i in range(len(tfidf)):
        find = False
        for j in target_col:
            if(tfidf[j][i]>0 and not find):
                find = True
                if(i < num_per_class):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\AKI\\'+str(i//10+1)+'.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                elif (i >= num_per_class and i < 2 * num_per_class):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\diabetes mellitus\\' + str(i // 10 + 1 - 25) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                elif (i >= 2 * num_per_class and i < 3 * num_per_class):
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\heart disease\\' + str(i // 10 + 1 - 50) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                else:
                    data = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\covid19\\' + str(i // 10 + 1 - 75) + '.csv'), encoding='utf-8', engine='python', usecols=['abstract'])
                #cal_sentence(i, data['abstract'][i % 10])
                for k in sent_tokenize(data['abstract'][i % 10]):  # j for every sentence
                    find_s = False
                    temp = []
                    for l in word_tokenize(k):  # k for every words
                        temp.append(l.lower())
                    tokens_tag = pos_tag(temp)
                    #print(tokens_tag)
                    for l in range(len(tokens_tag)):
                        temp[l] = temp[l] + '%' + tokens_tag[l][1].lower()
                        for m in target_col:
                            if (m == temp[l] and not find_s):
                                find_s = True
                                score = tfidf[m][i]
                                double = False
                                if (type == "aki" and i < num_per_class):
                                    double = True
                                elif (type == "diabetes_mellitus" and i >= num_per_class and i < 2 * num_per_class):
                                    double = True
                                elif (type == "heart_disease" and i >= 2 * num_per_class and i < 3 * num_per_class):
                                    double = True
                                elif (type == "covid19" and i >= 3 * num_per_class and i < 4 * num_per_class):
                                    double = True
                                type_list.append([i//250, double])
                                target_sentence_bm25.append(k)
                                #cal_bm25(temp[l].split('%')[0], k)

                                replaced_s = k.replace(" " + temp[l].split('%')[0], " <mark>" + temp[l].split('%')[0] + "</mark>")

                                # Replace the title keyword
                                insensitive = re.compile(re.escape(temp[l].split('%')[0]+" "), re.IGNORECASE)
                                replaced_s = insensitive.sub("<mark>" + temp[l].split('%')[0].capitalize() + "</mark> ", replaced_s)

                                target_sentence.append([round(score, 2), replaced_s])

    score_bm25 = cal_bm25(target_col, target_sentence_bm25)
    for i in range(len(target_sentence)):
        if(score_bm25[i]==0):
            score_bm25[i] = 0.1 + random.randint(0, 10)/100.0 # if bm25_score = 0, random(0.1x)

        if(type_list[i][1]):# double by theme
            target_sentence[i][0] = 2 * target_sentence[i][0]
            score_bm25[i] = 2 * score_bm25[i]
        target_sentence[i].append(round(score_bm25[i], 2)) # [tf_score, abstract, bm25_score]

        if(type_list[i][0] == 0): target_sentence[i].append('AKI')
        elif (type_list[i][0] == 1): target_sentence[i].append('Diabetes Mellitus')
        elif (type_list[i][0] == 2): target_sentence[i].append('Heart Disease')
        elif (type_list[i][0] == 3): target_sentence[i].append('COVID 19') # [tf_score, abstract, bm25_score, type]
        #print(target_sentence[i])

    if(tf_form=='tfidf'):
        target_sentence = sorted(target_sentence, reverse = True)
    else:
        target_sentence = sorted(target_sentence, key = lambda x: x[2], reverse = True)
    return render(request, "result.html", {'target_sentence': target_sentence[:50]})



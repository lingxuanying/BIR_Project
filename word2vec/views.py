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
from gensim.models import Word2Vec
import random
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

input_word = ""
target_col = []
target_sentence = []
model = Word2Vec.load('word2vec\\static\\assets\\docs\\word2vec.model')

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

def read_data(index):
    num_per_class = 250
    if (index < num_per_class):
        data = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\AKI\\' + str(index // 10 + 1) + '.csv'), encoding='utf-8', engine='python')

    elif (index >= num_per_class and index < 2 * num_per_class):
        data = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\diabetes mellitus\\' + str(index // 10 + 1 - 25) + '.csv'), encoding='utf-8', engine='python')

    elif (index >= 2 * num_per_class and index < 3 * num_per_class):
        data = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\heart disease\\' + str(index // 10 + 1 - 50) + '.csv'), encoding='utf-8', engine='python')

    else:
        data = pd.read_csv(os.path.join(settings.BASE_DIR,'word2vec\\static\\assets\\docs\\covid19\\' + str(index // 10 + 1 - 75) + '.csv'),encoding='utf-8', engine='python')

    return data

def search(request, tf_form, text, type):
    global input_word
    global target_col
    global target_sentence
    ps = PorterStemmer()

    if(text!='none'):
        input_word = text.split(' ')
        for i in range(len(input_word)):
            input_word[i] = ps.stem(input_word[i])

    tfidf = pd.read_csv(os.path.join(settings.BASE_DIR, 'word2vec\\static\\assets\\docs\\tfidf (2).csv'), encoding='utf-8', engine='python')
    #print(input_word)
    target_col = []
    for word in input_word:
        for i in list(tfidf.columns):
            if(ps.stem(i.split('%')[0])==word):
                target_col.append(i)
    #print(target_col)

    num_per_class = 250
    target_sentence = []
    target_sentence_bm25 = []
    type_list = []
    for i in range(len(tfidf)):
        find = False
        data = read_data(i)
        score = 0
        for j in target_col:
            score = score + tfidf[j][i]
            if(tfidf[j][i]>0 and not find):
                find = True
                #cal_sentence(i, data['abstract'][i % 10])
                double = False
                if (type == "aki" and i < num_per_class):
                    double = True
                elif (type == "diabetes_mellitus" and i >= num_per_class and i < 2 * num_per_class):
                    double = True
                elif (type == "heart_disease" and i >= 2 * num_per_class and i < 3 * num_per_class):
                    double = True
                elif (type == "covid19" and i >= 3 * num_per_class and i < 4 * num_per_class):
                    double = True
                type_list.append([i // 250, double])
                target_sentence_bm25.append(data['abstract'][i % 10])  # [abstract]
                target_sentence.append([round(score, 2), i, data['title'][i % 10]])


    score_bm25 = cal_bm25(target_col, target_sentence_bm25)
    for i in range(len(target_sentence)):
        if(score_bm25[i]==0):
            score_bm25[i] = 0.1 + random.randint(0, 10)/100.0 # if bm25_score = 0, random(0.1x)

        if(type_list[i][1]):# double by theme
            target_sentence[i][0] = 2 * target_sentence[i][0]
            score_bm25[i] = 2 * score_bm25[i]
        target_sentence[i].append(round(score_bm25[i], 2)) # [tf_score, index, abstract, bm25_score]

        if(type_list[i][0] == 0): target_sentence[i].append('AKI')
        elif (type_list[i][0] == 1): target_sentence[i].append('Diabetes Mellitus')
        elif (type_list[i][0] == 2): target_sentence[i].append('Heart Disease')
        elif (type_list[i][0] == 3): target_sentence[i].append('COVID 19') # [tf_score, index, abstract, bm25_score, type]
        #print(target_sentence[i])

    if(tf_form=='tfidf'):
        target_sentence = sorted(target_sentence, reverse = True)
    else:
        target_sentence = sorted(target_sentence, key = lambda x: x[3], reverse = True)
    return render(request, "result.html", {'target_sentence': target_sentence[:50]})

def rank_article(index):
    global target_sentence

    current_word_embedding = [0] * 250
    word_num = 0
    loss = []

    # calculate current article's word embedding
    data = read_data(index)
    for j in sent_tokenize(data['abstract'][index % 10]):  # j for every sentence
        # calculate word2vec
        for k in word_tokenize(j):  # k for every words
            try:
                current_word_embedding = current_word_embedding + model[k.lower()]
                word_num = word_num + 1
            except:
                continue
    current_word_embedding = [x / word_num for x in current_word_embedding]

    # ||top50 articles word embedding -  current article's word embedding||1
    for i in range(min(len(target_sentence), 50)):
        data = read_data(target_sentence[i][1])
        word_embedding = [0] * 250
        word_num = 0
        for j in sent_tokenize(data['abstract'][target_sentence[i][1] % 10]):  # j for every sentence
            # calculate word2vec
            for k in word_tokenize(j):  # k for every words
                try:
                    word_embedding = word_embedding + model[k.lower()]
                    word_num = word_num + 1
                except:
                    continue
        word_embedding = [x / word_num for x in word_embedding]
        loss.append(np.linalg.norm((np.array(word_embedding) - np.array(current_word_embedding)), ord=1))

    recommend = []
    loss[np.argmin(loss)] = 100000  # set a big num to delete current article in recommend list
    for i in range(5):
        #print(loss)
        recommend.append(np.argmin(loss))
        loss[np.argmin(loss)] = 100000 # set a big num to delete minimum number
    return recommend


def article(request, index):
    global model
    global target_col
    global target_sentence
    index = int(index)
    recom_article = []

    # read title, abstract
    data = read_data(index)

    # mark
    text = ""
    for i in sent_tokenize(data['abstract'][index % 10]):  # i for every sentence
        replaced_s = i
        for j in target_col:
            replaced_s = replaced_s.replace(" " + j.split('%')[0], " <mark>" + j.split('%')[0] + "</mark>")

            # Replace the title keyword
            insensitive = re.compile(re.escape(j.split('%')[0] + " "), re.IGNORECASE)
            replaced_s = insensitive.sub("<mark>" + j.split('%')[0].capitalize() + "</mark> ", replaced_s)
        text = text + replaced_s
    print(text)

    # random recommend
    recom_index = rank_article(index)
    for i in recom_index:
        recom_data = read_data(target_sentence[i][1])
        recom_article.append([target_sentence[i][1], recom_data['title'][target_sentence[i][1] % 10]])

    return render(request, "article.html", {'title': data['title'][index % 10], 'abstract': text, 'recom_data': recom_article}) #必须用这个return


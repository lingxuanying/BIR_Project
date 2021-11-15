from django.shortcuts import render
import gensim
from gensim.models import Word2Vec
import os
from django.conf import settings
import matplotlib.pyplot as plt
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
import pandas as pd
import re



# Create your views here.
def index(request):
    return render(request, "index.html") #必须用这个return

def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df

def search(request, text):
    model1 = Word2Vec.load(os.path.join(settings.BASE_DIR, 'word2vec.model'))
    word = re.sub(r'[^\w\s]', '', text.lower().strip())
    fig = plt.figure()
    ## word embedding
    tot_words = [word] + [tupla[0] for tupla in
                          model1.wv.most_similar(word, topn=10)]
    print(model1.wv.most_similar(word, topn=10))
    X = model1.wv[tot_words]
    ## pca to reduce dimensionality from 300 to 3
    pca = manifold.TSNE(perplexity=40, n_components=3, init='pca', random_state=20)
    X = pca.fit_transform(X)
    ## create dtf
    dtf_ = pd.DataFrame(X, index=tot_words, columns=["x", "y", "z"])
    dtf_["input"] = 0
    dtf_["input"].iloc[0:1] = 1
    ## plot 3d
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dtf_[dtf_["input"] == 0]['x'],
               dtf_[dtf_["input"] == 0]['y'],
               dtf_[dtf_["input"] == 0]['z'], c="black")
    ax.scatter(dtf_[dtf_["input"] == 1]['x'],
               dtf_[dtf_["input"] == 1]['y'],
               dtf_[dtf_["input"] == 1]['z'], c="red")
    ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[],
           yticklabels=[], zticklabels=[])
    for label, row in dtf_[["x", "y", "z"]].iterrows():
        x, y, z = row
        ax.text(x, y, z, s=label)

    plt.savefig('./word2vec/static/assets/test.png')
    top10 = []
    for tupla in model1.wv.most_similar(word, topn=10):
        top10.append(tupla[0])
    return render(request, "result.html", {'text': text, 'top10':top10})

def statistics(request):
    return render(request, "statistics.html") #必须用这个return

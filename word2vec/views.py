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
import plotly.graph_objs as go
from sklearn.decomposition import PCA




# Create your views here.
def index(request):
    return render(request, "index.html") #必须用这个return

def most_similar(w2v_model, words, topn=5):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df


def append_list(sim_words, words):
    list_of_words = []

    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)

    return list_of_words


def display_pca_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model.wv[w] for w in words])

    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :3]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    for i in range(len(user_input)):
        trace = go.Scatter3d(
            x=three_dim[count:count + topn, 0],
            y=three_dim[count:count + topn, 1],
            z=three_dim[count:count + topn, 2],
            text=words[count:count + topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter3d(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        z=three_dim[count:, 2],
        text=words[count:],
        name='input words',
        textposition="top center",
        textfont_size=20,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    data.append(trace_input)

    # Configure the layout

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    #plot_figure.show()
    return plot_figure

def display_pca_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model.wv[w] for w in words])

    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :2]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    for i in range(len(user_input)):
        trace = go.Scatter(
            x=three_dim[count:count + topn, 0],
            y=three_dim[count:count + topn, 1],
            text=words[count:count + topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        text=words[count:],
        name='input words',
        textposition="top center",
        textfont_size=20,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    data.append(trace_input)

    # Configure the layout

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    #plot_figure.show()
    return plot_figure

def search(request, text):
    model1 = Word2Vec.load(os.path.join(settings.BASE_DIR, 'word2vec.model'))
    input_word = text
    user_input = [x.strip() for x in input_word.split(' ')]
    result_word = []
    top = []
    for words in user_input:
        sim_words = model1.wv.most_similar(words, topn=5)
        sim_words = append_list(sim_words, words)

        result_word.extend(sim_words)
        top5 = []
        for tupla in model1.wv.most_similar(words, topn=5):
            top5.append(tupla[0])
        top.append(top5)
    top = np.array(top).T.tolist()
    similar_word = [word[0] for word in result_word]
    similarity = [word[1] for word in result_word]
    similar_word.extend(user_input)
    labels = [word[2] for word in result_word]
    label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]
    figure = display_pca_scatterplot_3D(model1, user_input, similar_word, labels, color_map)
    graph = figure.to_html()
    figure = display_pca_scatterplot_2D(model1, user_input, similar_word, labels, color_map)
    graph2D = figure.to_html()
    #plt_div = plotly.offline.plot(figure, output_type='div')
    return render(request, "result.html", {'graph': graph, 'top': top, 'user_input': user_input, 'graph2D': graph2D})


'''
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
    '''
def statistics(request):
    return render(request, "statistics.html") #必须用这个return

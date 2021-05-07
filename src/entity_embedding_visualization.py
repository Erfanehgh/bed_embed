import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import itertools
import pybedtools
from random import shuffle
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import argparse

# Read txt file

def data_preprocessing(path_embeded_document):
    document_embedding = pd.read_csv(path_embeded_document, header = None)
    
    document_embedding = (document_embedding[0].str.split('__label__', expand = True))
    document_embedding[1] = document_embedding[1].shift(1)
    document_embedding = document_embedding[5:].dropna()
    
    X = document_embedding[0].str.split(' ', expand = True)
    X = X[list(X)[0:-1]].astype(float)
    X = list(X.values)
    y = list(document_embedding[1])

    return X, y

def label_preprocessing(path_word_embedding, no_labels):

    labels = []
    label_vectors = []
    word_embedding = pd.read_csv(path_word_embedding, sep = '\t', header = None)
    vectors = word_embedding.tail(no_labels).reset_index()
    
    for l in range(no_labels):
        label_vectors.append((list(vectors.iloc[l])[2:]))
        labels.append(list(vectors.iloc[l])[1].replace('__label__',''))

    return label_vectors, labels

    

# This function reduce the dimension using umap and plot 
def visulization(input_data, labels, title = '', plot_type = 'umap', n_neighbours = 100, metric = 'euclidean', filename = '', plottitle = 'Label', output_folder = './'):

    n_labels=len(set(labels))
    
    np.random.seed(42)
    dp = 300
    
    if(plot_type == 'umap'):
        plot_model = umap.UMAP(metric=  metric, min_dist=0.01, n_components=2, n_neighbors = n_neighbours, random_state = 42)
    if(plot_type == 'tsne'):
        plot_model = TSNE(n_components = 2, perplexity=n_neighbours)
    if(plot_type == 'pca'):
        plot_model = PCA(n_components = 2)

    data = pd.DataFrame(plot_model.fit_transform(input_data)) 

    data = pd.DataFrame({'dim 1':data[0],
                            'dim 2':data[1],
                            title:[str(y1) for y1 in labels]})

    data = data.sort_values(by = title)
#     print(data)
    fig, ax = plt.subplots(figsize=(12,12))
    
    plate = sns.color_palette("husl", n_colors = n_labels)

#     plates = []
#     label_set = list(set(labels))
#     for i in range(len(plate)):
#         plates.extend([plate[i]]*len(data[data[title]==label_set[i]]))
        
#     shuffle(plate)

    
#     plt.scatter(x=data["dim 1"][0:-n_labels], y=data["dim 2"][0:-n_labels], hue=title, s = 50, marker = 'o', c = plates, edgecolor='black',
#                         linewidth=0.1, sizes=(100, 900),
# #                   data=data.sort_values(by = title),
#                 rasterized=True)
#     plt.scatter(x=data["dim 1"][-n_labels:], y=data["dim 2"][-n_labels:], hue=title, s = 400,  marker = '^', c = plates, sizes=(100, 900),
#                  rasterized=True, legend = None)

    sns.scatterplot(x=data["dim 1"][0:-n_labels], y=data["dim 2"][0:-n_labels], hue=title, s = 50,  palette = plate, sizes=(100, 900),
                  data=data.sort_values(by = title),rasterized=True)
    sns.scatterplot(x=data["dim 1"][-n_labels:], y=data["dim 2"][-n_labels:], hue=title, s = 400,  marker = '^', palette = plate, sizes=(100, 900),
                  data=data.sort_values(by = title),rasterized=True, legend = None)
    
    plt.legend(loc='upper left', fontsize =  15, markerscale=2, edgecolor = 'black')
    
    return fig


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_path", default=None, type=str,required=True, help="Path to the input file.",)

parser.add_argument("-emb", "--emb_path", default=None, type=str,
                        required=True,
                        help="Number of files to read.",)

parser.add_argument("-o", "--output", default='./', type=str,
                        required=True,
                        help="Path to output directory to store plots.",)

parser.add_argument("-plt", "--plot_type", default='', type=str,
                        required=True,
                        help="Type of the plot: [umap, t-sne, PCA]",)

parser.add_argument("-nn", "--no_neighbours", default=2, type=int,
                        required=True,
                        help="Number of neighbours for umap and t-sne plot.",)

parser.add_argument("-metric", "--metric", default='cosine', type=str,
                        required=True,
                        help="Distance metric for the umap plot.",)


args = parser.parse_args()

    
# path_document_embedding_train = './train_starspace_embed1574.txt'
# path_document_embedding_test = './test_starspace_embed1574.txt'
# path_word_embedding = './train_teststarspace1574.tsv'


path_document_embedding = args.input_path
path_word_embedding = args.emb_path
path_output = args.output
plot_type = args.plot_type
nn = args.no_neighbours
metric=args.metric


X, y = data_preprocessing(path_document_embedding)
X_label, y_label = label_preprocessing(path_word_embedding, len(set(y)))

X.extend(X_label)
y.extend(y_label)


fig = visulization(X, y, title = 'Cell', plot_type = plot_type, n_neighbours = nn, metric = metric, filename = '', plottitle = 'Label', output_folder = path_output)
fig.savefig(path_output+'cell_cosine.svg', format = 'svg')
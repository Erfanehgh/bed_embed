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



# Read txt file

def data_preprocessing(path_embeded_document):
    document_embedding = pd.read_csv(path_document_embedding_train, header = None)
    document_embedding = (document_embedding[0].str.split('hg', expand = True))
    document_embedding[1] = document_embedding[1].shift(1)
    document_embedding = document_embedding[document_embedding[1].isin(['19','38'])]
    X = []
    y = []


    for i in range(4, len(document_embedding)):
        if (i%2 == 0):
            y.append(np.int(document_embedding.iloc[i][1])/19-1)
            X.append([float(z) for z in document_embedding.iloc[i][0].split(' ')[0:-1]])

#     document_embedding = pd.read_csv(path_embeded_document, header = None)
#     document_embedding[[0,1, 2]] = (document_embedding[0].str.split('hg', expand = True))
    
    
#     X = []
#     y = []

#     for i in range(4, len(document_embedding)):
#         if (i%2 == 0):
#             if (document_embedding.iloc[i][1]==None):
#                 print(i)
#                 i+=1
#             if(('19' in str(document_embedding.iloc[i][2])) or ('38' in str(document_embedding.iloc[i][2])) ):
#                 y.append(np.int(document_embedding.iloc[i][2])/19-1)
#                 continue
#             y.append(np.int(document_embedding.iloc[i][1])/19-1)
#         else:
#             X.append([float(z) for z in document_embedding.iloc[i][0].split(' ')[0:-1]])
            
#     document_embedding = pd.read_csv(path_embeded_document, header = None)
#     document_embedding[[0,1]] = (document_embedding[0].str.split('hg', expand = True))
  
#     X = []
#     y = []

#     for i in range(4, len(document_embedding)):
#         if (i%2 == 0):
#             if (document_embedding.iloc[i][1]==None):
#                 print(i)
#                 i+=1
# #             if(('19' in str(document_embedding.iloc[i][2])) or ('38' in str(document_embedding.iloc[i][2])) ):
# #                 y.append(np.int(document_embedding.iloc[i][2])/19-1)
# #                 continue
#             y.append(np.int(document_embedding.iloc[i][1])/19-1)
#         else:
#             X.append([float(z) for z in document_embedding.iloc[i][0].split(' ')[0:-1]])

    return X, y

def label_preprocessing(path_word_embedding, nolabels):
    labels = []
    label_vectors = []
    word_embedding = pd.read_csv(path_word_embedding, sep = '\t', header = None)
    vectors = word_embedding.tail(nolabels).reset_index()
    
    label_vectors.append((list(vectors.iloc[0])[2:]))
    label_vectors.append((list(vectors.iloc[1])[2:]))
    labels.append('hg19')
    labels.append('hg38')
    return label_vectors, labels

    

# This function reduce the dimension using umap and plot 
def visulization(input_data, labels, title = '', plot_type = 'umap', n_neighbours = 100, metric = 'euclidean', filename = '', plottitle = 'Label', output_folder = './'):

    np.random.seed(42)
    dp = 300
    
    if(plot_type == 'umap'):
        plot_model = umap.UMAP(metric=  metric, min_dist=0.01, n_components=2, n_neighbors = n_neighbours, random_state = 42)
    if(plot_type == 'tsne'):
        plot_model = TSNE(n_components = 2, perplexity=n_neighbours)
    if(plot_type == 'pca'):
        plot_model = PCA(n_components = 2)

    data = pd.DataFrame(plot_model.fit_transform(input_data)) 
    print(Counter(labels))

    data = pd.DataFrame({'dim 1':data[0],
                            'dim 2':data[1],
                            title:[str(y1) for y1 in labels]})

    fig, ax = plt.subplots(figsize=(12,12))
    
    plate =(sns.color_palette("husl", n_colors=len(set(labels))))
    
    shuffle(plate)
    
    sns.scatterplot(x=data["dim 1"][0:-2], y=data["dim 2"][0:-2], hue=title, s =50,  palette = plate, sizes=(100, 900),
                  data=data.sort_values(by = title),rasterized=True)
    sns.scatterplot(x=data["dim 1"][-2:], y=data["dim 2"][-2:], hue=title, s = 400,  marker = '^', palette = plate, sizes=(100, 900),
                  data=data.sort_values(by = title),rasterized=True, legend = None)

    plt.legend(loc='upper left', fontsize =  15, markerscale=2, edgecolor = 'black')
    
    return fig


    
path_document_embedding_train = '/scratch/eg8qe/StarSpace/train_starspace_embed1574.txt'
path_document_embedding_test = '/scratch/eg8qe/StarSpace/test_starspace_embed1574.txt'
path_word_embedding = './train_teststarspace1574.tsv'



X, y = data_preprocessing(path_document_embedding_test)
X_label, y_label = label_preprocessing(path_word_embedding, 2)

X.extend(X_label)
y.extend(y_label)



nn = 50
fig = visulization(X, y, title = 'Assembly', plot_type = 'umap', n_neighbours = nn, metric = 'euclidean', filename = '', plottitle = 'Label', output_folder = './')
fig.savefig('assembly004_cosine_all.svg', format = 'svg')


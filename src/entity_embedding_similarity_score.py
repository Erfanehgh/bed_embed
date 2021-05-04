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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

# Read txt file

def data_preprocessing(path_embeded_document):
    document_embedding = pd.read_csv(path_embeded_document, header = None)
    print(len(document_embedding))
    document_embedding = (document_embedding[0].str.split('__label__', expand = True))
    document_embedding[1] = document_embedding[1].shift(1)
    document_embedding = document_embedding[5:].dropna()
    print(len(document_embedding))
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

    

def calculate_accuracy(X_query, X_label, y_label, y):
    tp = 0
    
    for i in range(len(X_query)):
        query = X_query[i]
        distance = []
        for j in range(len(X_label)):
            distance.append((cosine_similarity(np.array(query).reshape(1, -1), np.array(X_label[j]).reshape(1, -1)))[0][0])

        label = y_label[np.argmax(distance)]
        if(label==y[i]):
            tp+=1
    return (tp/(i+1))

def calculate_distance(X_files, X_labels, y_labels):
    X_files = np.array(X_files)
    X_labels = np.array(X_labels)
    distance_matrix = distance.cdist(X_files, X_labels, 'cosine')
    df_distance_matrix = pd.DataFrame(distance_matrix)
    df_distance_matrix.columns = y_label
    df_distance_matrix['file_id'] = df_distance_matrix.index
    file_distance = pd.melt(df_distance_matrix, id_vars= 'file_id', var_name='search_term', value_name='score')
    scaler = MinMaxScaler()
    file_distance['score'] = scaler.fit_transform(np.array(file_distance['score']).reshape(-1,1))
    return file_distance
    

    

path_document_embedding_train = './train_starspace_embed1574.txt'
path_document_embedding_test = './test_starspace_embed1574.txt'
path_word_embedding = './train_teststarspace1574.tsv'


X, y = data_preprocessing(path_document_embedding_train)
X_label, y_label = label_preprocessing(path_word_embedding, len(set(y)))

print(calculate_accuracy(X, X_label, y_label, y))
calculate_distance(X, X_label, y_label)
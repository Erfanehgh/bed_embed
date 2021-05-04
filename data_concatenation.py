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
import glob




def data_prepration(path_file_label):
#     print(path_file_label)
    path_file, label = path_file_label.split(',')
    if os.path.exists(path_file):
        try:
            df = pybedtools.BedTool(path_file)
            file_regions = universe.intersect(df, wa=True) 
            file_regions.columns = ['chrom', 'start', 'end']
            if(len(file_regions)==0):
                return ' '
            print(path_file_label)
            file_regions = (file_regions.to_dataframe().drop_duplicates())
            file_regions['region'] = file_regions['chrom'] + '_' + file_regions['start'].astype(str) + '_' + file_regions['end'].astype(str) 
            return ' '.join(list(file_regions['region']))+ ' __label__' + label
        except Exception:
                print('Error in reading file: ', path_file)
                return ' ' 
    else:
        return ' '
    
    
def split_train_test(documents, prop = 0.2, path_output = './'):
    
    shuffle(documents)
    train_files, test_files = train_test_split(documents, test_size = prop, random_state = 42)
    with open(path_output + 'train_documents{}.txt'.format(len(train_files)),'w') as input_file:
        input_file.write(' '.join(train_files))
    with open(path_output + 'test_documents{}.txt'.format(len(test_files)),'w') as input_file:
        input_file.write(' '.join(test_files))
    print(len(train_files), len(test_files))
        

        
trained_documents = []
for file in sorted(glob.glob('/scratch/eg8qe/StarSpace/trained_documents*file100*'))[0:10]:
    print(file)
    file1 = open(file, 'r')
    Lines = file1.readlines()
    for line in Lines:
        trained_documents.append(line)
    print(len(trained_documents))
    trained_documents = list(set(trained_documents))
    print(len(trained_documents))
    trained_documents.extend(pd.read_csv(file))


    
split_train_test(trained_documents, 0.2, '/scratch/eg8qe/StarSpace/')
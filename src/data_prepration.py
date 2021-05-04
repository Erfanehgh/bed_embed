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
    shuffle(trained_documents)
    train_files, test_files = train_test_split(documents, test_size = prop, random_state = 42)
    with open(path_output + 'train_documents{}.txt'.format(len(train_files)),'w') as input_file:
        input_file.write('\n'.join(train_files))
    with open(path_output + 'test_documents{}.txt'.format(len(test_files)),'w') as input_file:
        input_file.write('\n'.join(test_files))
        

        

no_files = 100
n_process = 20
tileLen = 1000
path_universe = './universe_tilelen{}.bed'.format(tileLen)
universe = pybedtools.BedTool(path_universe)

print(len(universe))
file_list = list(pd.read_csv('file_list_cell.txt', header = None, sep = ' ')[0])
shuffle(file_list)
for i in range(0, 100, no_files):
    print(i)
    
    trained_documents = []
    with Pool(n_process) as p:
        trained_documents = p.map(data_prepration, file_list[i:i+no_files])  
        p.close()
        p.join()
    print(len(trained_documents))
    print('Reading files done')

    while (' ' in trained_documents):
        trained_documents.remove(' ')
        
    print(len(trained_documents))

    with open('./documents_cell_chunk{}_file{}_tilelen{}.txt'.format(int(i/no_files), no_files, tileLen),'w') as input_file:
        input_file.write('\n'.join(trained_documents))
    input_file.close()
    
print(len(trained_documents))
split_train_test(trained_documents, 0.2, './')
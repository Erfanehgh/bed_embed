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
from ubiquerg import VersionInHelpParser
import argparse


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
#             print(path_file_label)
            file_regions = (file_regions.to_dataframe().drop_duplicates())
            file_regions['region'] = file_regions['chrom'] + '_' + file_regions['start'].astype(str) + '_' + file_regions['end'].astype(str) 
            return ' '.join(list(file_regions['region']))+ ' __label__' + label
        except Exception:
                print('Error in reading file: ', path_file)
                return ' ' 
    else:
        return ' '
        
def split_train_test(documents, prop = 0.2, path_output = './', meta = ''):
    shuffle(trained_documents)
    train_files, test_files = train_test_split(documents, test_size = prop, random_state = 42)
    with open(path_output + 'train_documents_{}.txt'.format(meta),'w') as input_file:
        input_file.write('\n'.join(train_files))
    with open(path_output + 'test_documents_{}.txt'.format(meta),'w') as input_file:
        input_file.write('\n'.join(test_files))
        

        

        



        


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_path", default=None, type=str,required=True, help="Path to input file.",)
parser.add_argument("-univ", "--univ_path", default=None, type=str,
                        required=True,
                        help="Path to universe file.",)

parser.add_argument("-nf", "--no_files", default=None, type=int,
                        required=True,
                        help="Number of files to read.",)

parser.add_argument("-o", "--output", default='./', type=str,
                        required=True,
                        help="Path to output directory to store file.",)

parser.add_argument("-meta", "--meta_label", default='', type=str,
                        required=True,
                        help="The meta data type",)

args = parser.parse_args()
# print(args.input)


# print(args.accumulate(args.integers))


# no_files = 10
# n_process = 20
tileLen = 1000
# path_universe = './universe/universe_tilelen1000.bed'
# path_input = './meta_data/file_list_cell.txt'
# path_output = './'



no_files = args.no_files
n_process = 20
path_universe = args.univ_path
path_input = args.input_path
path_output = args.output
meta_data=args.meta_label


universe = pybedtools.BedTool(path_universe)

print(len(universe))
file_list = list(pd.read_csv(path_input, header = None, sep = ' ')[0])
# shuffle(file_list)
# for i in range(0, 20, no_files):
#     print(i)
    
trained_documents = []
with Pool(n_process) as p:
    trained_documents = p.map(data_prepration, file_list[0:no_files])  
    p.close()
    p.join()
print(len(trained_documents))
print('Reading files done')

while (' ' in trained_documents):
    trained_documents.remove(' ')
        
print(len(trained_documents))

with open(path_output + 'documents_{}_file{}_tilelen{}.txt'.format(meta_data,no_files, tileLen),'w') as input_file:
    input_file.write('\n'.join(trained_documents))
input_file.close()
    
print(len(trained_documents))
split_train_test(trained_documents, 0.0001, path_output, meta_data)
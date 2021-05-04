import pandas as pd
import os
from random import shuffle
from multiprocessing import Pool
import itertools


def read_file_minmax(path_file):
    print(path_file)
    univ_minmax = pd.DataFrame()
    if os.path.exists(path_file):
        df = pd.read_csv(path_file, sep = '\t', header = None)[[0, 1, 2]].rename(columns={0:'Chromosome', 1:'Start_position', 2:'End_position'})
        univ = df[['Chromosome', 'Start_position', 'End_position']].sort_values(['Chromosome', 'Start_position', 'End_position'])
        univ_minmax = univ.groupby('Chromosome')['Start_position'].min().reset_index()
        univ_minmax['End_position'] = univ.groupby('Chromosome')['End_position'].max().reset_index()['End_position']
    return univ_minmax 

def tiling(ch):
    tile_chrom = []
    row = univ_minmax[univ_minmax.Chromosome == ch]
    for i in range(list(row['Start_position'])[0], list(row['End_position'])[0], tileLen):
        tile_chrom.append([ch, i, i + tileLen])
    return tile_chrom


tileLen = 1000
n_process = 20
no_of_files = 200


file_list = (list(pd.read_csv('hg19files.txt', header = None)[0]))
file_list.extend(list(pd.read_csv('hg38files.txt', header = None)[0]))
print(len(file_list))


univ_minmax_all = pd.DataFrame(columns = ['Chromosome', 'Start_position', 'End_position'])

with Pool(n_process) as p:
    univ_minmax_all = p.map(read_file_minmax, file_list[0:no_of_files])
    p.close()
    p.join()
    
print('Reading Done')   
univ_minmax_all = pd.concat(univ_minmax_all)


univ_minmax = pd.DataFrame()
univ_minmax = univ_minmax_all.groupby('Chromosome')['Start_position'].min().reset_index()
univ_minmax['End_position'] = univ_minmax_all.groupby('Chromosome')['End_position'].max().reset_index()['End_position']

univ_minmax['len'] = univ_minmax['End_position'] - univ_minmax['Start_position']
# univ_minmax['noTiles']=univ_minmax['len']/1000

tile_universe_all = []

list_chrom = list(set(univ_minmax.Chromosome))

with Pool(n_process) as p:
    tiles = list(itertools.chain.from_iterable(p.map(tiling, list_chrom)))
    p.close()
    p.join()
    

tile_universe = pd.DataFrame.from_records(tiles).sort_values(by = [0, 1, 2])
tile_universe.to_csv('universe_tilelen{}.bed'.format(tileLen), sep = '\t', header = None, index = None)



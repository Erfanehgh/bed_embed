source activate py3.7
module load gcc/9.2.0 boost


meta='target'

# Data prepration 
path_input='./meta_data/file_list_'$meta'.txt'
path_universe='/home/eg8qe/Desktop/gitHub/region2vec/StarSpaceEmbedding/universe_tilelen1000.bed'
path_output='./'
no_files=10
path_data='/project/shefflab/data/encode/'

# StarSpace Training

path_embedded_documents_train=$path_output'train_starspace_embed_'$meta'.txt'
path_embedded_labels=$path_output'starspace_model_'$meta'.tsv'


./Starspace/starspace train -trainFile $path_output'documents_file'$no_files'_'$meta'.txt' -model $path_output/starspace_model_$meta -trainMode 0 -dim 50 -epoch 10 -negSearchLimit 5 -thread 20 -lr 0.001





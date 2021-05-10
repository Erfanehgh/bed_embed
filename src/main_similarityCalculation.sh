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



# Document Embedding

./Starspace/embed_doc $path_output/starspace_model_$meta $path_output/documents_file$no_files\_$meta.txt > $path_embedded_documents_train

#Calculate similarity score and accuracy 

path_output_similarity='./similarity_score/'
python ./src/entity_embedding_similarity_score.py -i $path_embedded_documents_train -emb $path_embedded_labels -o $path_output_similarity -meta $meta -mode train -filelist $path_input -nf $no_files
# python ./src/entity_embedding_similarity_score.py -i $path_embedded_documents_test -emb $path_embedded_labels -o $path_output_similarity -meta $meta -mode test







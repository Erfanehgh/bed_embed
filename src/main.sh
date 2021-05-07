source activate py3.7
module load gcc/9.2.0 boost


meta='target'

# Data prepration 
path_input='./meta_data/file_list_'$meta'.txt'
path_universe='/home/eg8qe/Desktop/gitHub/region2vec/StarSpaceEmbedding/universe_tilelen1000.bed'
path_output='/scratch/eg8qe/StarSpace/testdocuments/'
no_files=400



path_embedded_documents_train='/scratch/eg8qe/StarSpace/testdocuments/train_starspace_embed_'$meta'.txt'
path_embedded_documents_test='/scratch/eg8qe/StarSpace/testdocuments/test_starspace_embed_'$meta'.txt'
path_embedded_labels='/scratch/eg8qe/StarSpace/testdocuments/starspace_model_'$meta'.tsv'


# Visualization parameters
path_output_plot='./plots/'
plot_type='umap'
nn=50
metric='cosine'

# Calculate Similarity

path_output_similarity='./similarity_score/'


python ./src/data_prepration.py -i $path_input -univ $path_universe -nf $no_files -o $path_output -meta $meta

./Starspace/starspace train -trainFile $path_output'train_documents_'$meta'.txt' -model $path_output/starspace_model_$meta -trainMode 0 -dim 50 -epoch 10 -negSearchLimit 5 -thread 20 -lr 0.001

./Starspace/embed_doc $path_output/starspace_model_$meta $path_output/train_documents_$meta.txt > $path_output/train_starspace_embed_$meta.txt

./Starspace/embed_doc $path_output/starspace_model_$meta $path_output/test_documents_$meta.txt > $path_output/test_starspace_embed_$meta.txt

python ./src/entity_embedding_visualization.py -i $path_embedded_documents_train -emb $path_embedded_labels -o $path_output_plot -plt $plot_type -nn $nn -metric $metric -meta $meta -mode train
python ./src/entity_embedding_visualization.py -i $path_embedded_documents_test -emb $path_embedded_labels -o $path_output_plot -plt $plot_type -nn 10 -metric $metric -meta $meta -mode test

python ./src/entity_embedding_similarity_score.py -i $path_embedded_documents_train -emb $path_embedded_labels -o $path_output_similarity -meta $meta -mode train
python ./src/entity_embedding_similarity_score.py -i $path_embedded_documents_test -emb $path_embedded_labels -o $path_output_similarity -meta $meta -mode test







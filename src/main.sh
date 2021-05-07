source activate py3.7
module load gcc/9.2.0 boost


path_input='./meta_data/file_list_cell.txt'
path_universe='./universe/universe_tilelen1000.bed'
path_output='./'
no_files=10



path_embedded_documents='./train_starspace_embed1574.txt'
path_embedded_labels='./train_teststarspace1574.tsv'
path_output_plot='./plots/'

plot_type='umap'
nn=5
metric='cosine'

path_output_similarity='./similarity_score/'


python ./src/data_prepration.py -i $path_input -univ $path_universe -nf $no_files -o $path_output
./Starspace/starspace train -trainFile $path_output'train_documents8.txt' -model ./train_teststarspace1574 -trainMode 0 -dim 50 -epoch 5 -negSearchLimit 5 -thread 20 -lr 0.001
./Starspace/embed_doc ./train_teststarspace1574 ./train_documents8.txt > ./train_starspace_embed1574.txt
./Starspace/embed_doc ./train_teststarspace1574 ./test_documents2.txt > ./test_starspace_embed1574.txt

python ./src/entity_embedding_visualization.py -i $path_embedded_documents -emb $path_embedded_labels -o $path_output_plot -plt $plot_type -nn $nn -metric $metric

python ./src/entity_embedding_similarity_score.py -i $path_embedded_documents -emb $path_embedded_labels -o $path_output_similarity 








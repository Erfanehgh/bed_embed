path_ssModel='/scratch/eg8qe/StarSpace'
module load gcc/9.2.0 boost




python data_prepration.py 

./Starspace/starspace train -trainFile '/scratch/eg8qe/StarSpace/train_documents1582.txt' -model ./train_teststarspace1574 -trainMode 0 -dim 50 -epoch 5 -negSearchLimit 5 -thread 20 -lr 0.001
./Starspace/embed_doc ./train_teststarspace1574 /scratch/eg8qe/StarSpace/train_documents1582.txt > ./train_starspace_embed1574.txt
./Starspace/embed_doc ./train_teststarspace1574 /scratch/eg8qe/StarSpace/test_documents396.txt > ./test_starspace_embed1574.txt

python entity_embedding_visualization.py 
python entity_embedding_accuracy.py 







source activate py3.7
module load gcc/9.2.0 boost


meta='target'

# Data prepration 
path_input='./meta_data/file_list_'$meta'.txt'
path_universe='./universe/universe_tilelen1000.bed'
path_output='./'
no_files=10
path_data='/project/shefflab/data/encode/'

python ./src/data_prepration.py -i $path_input -path_data $path_data -univ $path_universe -nf $no_files -o $path_output -meta $meta








#!/bin/bash

export DGLBACKEND="pytorch"

#--Definition of variables--------

# trained model's result directory
result_dir="/home/miniconda-user/workspace/fPLG/results/BindingDB_AlphaDrug_addTanky_standardized_scaled_log_IC50_QED_fast_Pretrain_0227"
# pdb directory (test data)
pdb_dir="/home/miniconda-user/workspace/fPLG/PDB/pdb_tankyrase"
# condition table for generating ligands
condition="/home/miniconda-user/workspace/fPLG/data/condition_table.csv"

#----------------------------------

# test
nohup python3 test.py ${result_dir}'/input_data/params.yml' ${pdb_dir} ${condition} --gpu 9 --n_jobs 24 >> $result_dir'/test.log' &


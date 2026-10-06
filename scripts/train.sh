#!/bin/bash

export DGLBACKEND="pytorch"

#--Definition of variables--------

# result directory
result_dir="/home/miniconda-user/workspace/fPLG/results/BindingDB_AlphaDrug_addTanky_standardized_scaled_log_IC50_QED_fast_0227"

# pretrained fPLG model (option)
pretrained_fplg=""

#----------------------------------

# train
nohup python3 train.py ${result_dir}'/input_data/params.yml' --pretrained_fplg $pretrained_fplg --gpus 0 1 2 3 --n_jobs 24 --save_interval 10 --valid --master_port 12355 > $result_dir'/train.log' &


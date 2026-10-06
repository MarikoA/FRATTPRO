#!/bin/bash

export DGLBACKEND="pytorch"

#--Definition of variables--------

# training data
data_path="/home/miniconda-user/workspace/fPLG/data/BindingDB_AlphaDrug_addTanky_standardized.csv"

# esm models (https://github.com/facebookresearch/esm?tab=readme-ov-file#available-models)
esm="esm2_t33_650M_UR50D"

# pretrained FRATTVAE model
pretrained_model="/home/miniconda-user/workspace/FRATTVAE/results/ChEMBL_DB_tg_standardized_struct"

#----------------------------------

# preprocessing
nohup python preprocessing.py $data_path \
                                $esm \
                                --gpu 5 \
                                --pretrained_model $pretrained_model \
                                --n_jobs 24 \
                                --seed 0 \
                                --maxDepth 32 \
                                --maxDegree 16 \
                                --minSize 1 \
                                --standardized \
                                --epoch 100 \
                                --batch_size 2048 \
                                --property scaled_log_IC50:1 \
                                --property QED:1 \
                                --lr 0.0001 >> preprocessing_fast.log &



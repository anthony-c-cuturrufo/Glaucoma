#!/bin/bash

declare -A configs

configs=(
    # ["exp01"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name SEResNet152 --dropout 0  --batch_size 11 --lr 1e-5" 
    # ["exp02"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0  --batch_size 11 --lr 1e-5"
    # ["exp03"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0  --batch_size 11 --weight_decay 1e-5 --lr 1e-5"
    # ["exp04"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0  --batch_size 6  --lr 1e-5 --precompute True"


    # ["exp07"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext50 --dropout 0  --batch_size 6 --lr 1e-5 --precompute True"
    # scai3 ["exp05"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext50 --dropout 0  --batch_size 6 --lr 1e-5 --imbalance_factor 2 --use_focal_loss True"
    # scai4 ["exp07"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0  --batch_size 10 --lr 1e-5 --imbalance_factor 2 --use_focal_loss True"
    ["exp08"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0  --batch_size 10 --lr 1e-5 --imbalance_factor 3 --use_focal_loss True"
    # ["exp02"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name MedicalNet --dropout 0.3  --batch_size 4 --lr 1e-5"
    # ["exp02"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name MedicalNet --dropout 0.5  --batch_size 4 --lr 1e-5 --epochs 700"
    # ["exp02"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0.1  --batch_size 11 --lr 1e-5"
    # ["exp02"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0  --batch_size 11 --lr 1e-5"
    # ["exp07"]="--dataset local_database9_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout .5  --batch_size 11 --lr 1e-5"


    # ["exp04"]="--dataset local_database8_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout .2  --batch_size 11 --lr 1e-5"
    # ["exp05"]="--cuda cuda:0 --dataset local_database8_Macular_SubMRN_v4.csv --model_name ResNext121 --dropout 0  --batch_size 5"
)

for exp in "${!configs[@]}"; do
    screen -S $exp -d -m bash -c "
    run_experiment() {
        source ~/.bashrc
        conda activate glauconda2
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \$1
    }
    run_experiment '${configs[$exp]}' > ${exp}.log 2>&1
    " 
done

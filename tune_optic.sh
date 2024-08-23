#!/bin/bash

declare -A configs

configs=(
    # ["resnext7"]="--config cfgs/cfg_resnext7.yml"
    # ["resnext8"]="--config cfgs/cfg_resnext8.yml"
    # ["resnext9"]="--config cfgs/cfg_resnext9.yml"
    # ["resnext10"]="--config cfgs/cfg_resnext10.yml"
    # ["resnext50"]="--config cfgs/cfg_resnext50.yml"
    # ["vit"]="--config cfgs/cfg_vit.yml"
    # ["dualViT"]="--config cfgs/cfg_dual_vit.yml"
    # ["3DCNN"]="--config cfgs/cfg_3DCNN.yml"
    # ["resnet10"]="--config cfgs/cfg_resnet10.yml"
    # ["resnet50"]="--config cfgs/cfg_resnet50.yml"
    # ["nilay"]="--config cfgs/cfg_nilay.yml"
    ["hiroshi"]="--config cfgs/cfg_hiroshi.yml"
    ["hiroshi2"]="--config cfgs/cfg_hiroshi2.yml"
    ["hiroshi3"]="--config cfgs/cfg_hiroshi3.yml"
    # ["hiroshi4"]="--config cfgs/cfg_hiroshi4.yml"


)

for exp in "${!configs[@]}"; do
    screen -S $exp -d -m bash -c "
    run_experiment() {
        source ~/.bashrc
        conda activate glauconda2
        python train_lit2.py fit \$1
    }
    run_experiment '${configs[$exp]}' > logs/${exp}.log 2>&1
    "
done

#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")


config=configs/cityscapes/segformerb2_769x_240e_lr0.005_1x8.yaml
config2=configs/cityscapes/r50_dct_64_769x_240e_lr0.005_1x8.yaml
labeled_id_path=partitions/cityscapes/$3/labeled.txt
unlabeled_id_path=partitions/cityscapes/$3/unlabeled.txt
save_path=exp/pascal/diverse_cotraining/2cps/$3

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    cotraining_rgb_dct.py --thr 0.0 \
    --config=$config --config2=$config2 --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt
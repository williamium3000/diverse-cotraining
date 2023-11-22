#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal/segformerb2_4x4.yaml
config2=configs/pascal/r50_dct_4x4.yaml
config3=configs/pascal/r50_4x4.yaml
labeled_id_path=partitions/pascal/$3/labeled.txt
unlabeled_id_path=partitions/pascal/$3/unlabeled.txt
save_path=exp/pascal/diverse_cotraining/3cps_321x_$4/$3

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    diverse_cotraining_3cps.py --thr $4 \
    --config=$config --config2=$config2 --config3=$config3 --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt
#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal/segformerb2_4x4.yaml
config2=configs/pascal/r50_dct_4x4.yaml
labeled_id_path=partitions/pascal/$3/labeled.txt
unlabeled_id_path=partitions/pascal/$3/unlabeled.txt
save_path=exp/pascal/diverse_cotraining/2cps_321x_$4/$3

mkdir -p $save_path


srun  -p partition -N 1 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=test --mem-per-cpu=32GB --cpus-per-task=5 \
python diverse_cotraining.py --thr $4 \
    --config=$config --config2=$config2 --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 \
    2>&1 | tee $save_path/$now.txt
#!/usr/bin/env bash
model_name=main
data_dir=/export/ILSVRC
checkpoint_name=outputs/${model_name}/checkpoint_0199.pth.tar
if [ -f "${checkpoint_name}" ]; then
    echo "${checkpoint_name} exist, skip pre-training"
else
    python main_moco.py \
    -a resnet50 \
    --lr 0.03 \
    --batch-size 256 \
    --dist-url 'tcp://localhost:10001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
    --mlp --moco-t 0.2 --aug-plus --cos \
    --model_version MoCoUnlimitedKeysDefault \
    --jcl_lambda 10.0 \
    --jcl 1 \
    --k_crops 5 \
    --epochs 200 \
    --out_folder outputs/${model_name} \
    "${data_dir}"
fi

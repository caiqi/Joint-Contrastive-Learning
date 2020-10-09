#!/usr/bin/env bash
model_name=main
data_dir=/export/ILSVRC
checkpoint_name=outputs/${model_name}/checkpoint_0199.pth.tar

python main_lincls.py \
-a resnet50 \
--lr 30.0 \
--batch-size 256 \
--dataset imagenet \
--pretrained ${checkpoint_name} \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--out_folder outputs/${model_name}_linear_run1 \
--epochs 100 \
"${data_dir}"

# Joint Contrastive Learning with Infinite Possibilities
This is the implementation of '**Joint Contrastive Learning with Infinite Possibilities**' [NeurIPS 2020]. The original paper can be found at https://arxiv.org/abs/2009.14776 .

## Requirements
* torch
* torchvision

## Datasets
* Download the ImageNet dataset from [http://www.image-net.org](http://www.image-net.org). 
* Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). 

### Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model, run:
```
bash scripts/main_pretrain.sh
```

### Evaluation of linear classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights, run:
```
bash scripts/main_lincls.sh
```

### Models

Our pre-trained ResNet-50 models can be downloaded from [ResNet-50](https://github.com/anonymouszyx/JCL/releases/download/v1/checkpoint_0199.pth.tar).


## Citation
If you find this code or model useful for your research, please cite our paper:

    @inproceedings{cai2020joint,
      title={Joint Contrastive Learning with Infinite Possibilities},
      author={Cai, Qi and Wang, Yu and Pan, Yingwei and Yao, Ting and Mei, Tao},
      booktitle={Advances in Neural Information Processing Systems},
      year={2020}
    }

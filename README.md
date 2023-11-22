# Diverse Co-training

Official PyTorch implementation of ICCV 2023 paper **"Diverse Cotraining Makes Strong Semi-Supervised Segmentor"**.

***Abstract.***
> Deep co-training has been introduced to semi-supervised segmentation and achieves impressive results, yet few studies have explored the working mechanism behind it. In this work, we revisit the core assumption that supports co-training: multiple compatible and conditionally independent views. By theoretically deriving the generalization upper bound, we prove the prediction similarity between two models negatively impacts the model's generalization ability. However, most current co-training models are tightly coupled together and violate this assumption. Such coupling leads to the homogenization of networks and confirmation bias which consequently limits the performance. To this end, we explore different dimensions of co-training and systematically increase the diversity from the aspects of input domains, different augmentations and model architectures to counteract homogenization. Our \textit{Diverse Co-training} outperforms the state-of-the-art (SOTA) methods by a large margin across different evaluation protocols on the Pascal and Cityscapes. For example. we achieve the best mIoU of 76.2\%, 77.7\% and 80.2\% on Pascal with only 92, 183 and 366 labeled images, surpassing the previous best results by more than 5\%.

## Results

### Pascal
#### labeled data sampled from high-quality training set
Labeled images are sampled from the **original high-quality** training set. Results are obtained by DeepLabv3+ with ResNet-101 and SegFormer-b3.

| Method | Resolution | 1/115 (92) | 1/57 (183) | 1/28 (366) | 1/14 (732) | 1/7 (1464) |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: | :-------: |
| SupOnly | 321x321 | 44.4 | 54.0 | 63.4 | 67.2 | 71.8 |
| ReCo | 321x321 | 64.8 | 72.0 | 73.1 | 74.7 | - |
| ST++ | 321x321 | 65.2 | 71.0 | 74.6 | 77.3 | 79.1 |
| **Ours(2-cps)** | 321x321 | *74.8* | **77.6** | *79.5* | *80.3* | **81.7** |
| **Ours(3-cps)**| 321x321 | **75.4** | *76.8* | **79.6** | **80.4** | *81.6* |
| SupOnly | 513x513 | 42.3 | 56.6 | 64.2 | 68.1 | 72.0 |
| U<sup>2</sup>PL | 512x512 | 68.0 | 69.2 | 73.7 | 76.2 | 79.5 |
| PS-MT | 512x512 | 65.8 | 69.6 | 76.6 | 78.4 | 80.0  |
| **Ours(2-cps)**| 513x513 | **76.2** | *76.6* | **80.2** | *80.8* | *81.9* |
| **Ours(3-cps)**| 513x513 | *75.7* | **77.7** | *80.1* | **80.9** | **82.0** |
#### labeled data sampled from blened training set
Labeled images are sampled from the **blened** training set. 
Results obtained by DeepLabv3+ with ResNet-50 and SegFormer-b2.

| Method | Resolution | 1/32 (331) | 1/16 (662) | 1/8 (1323) | 1/4 (2646) |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: |
| SupOnly | 321x321 | 55.8 | 60.3 | 66.8 | 71.3 |
| ST++ | 321x321 | - | 72.6 | 74.4 | 75.4 |
| **Ours(2-cps)** | 321x321 | **75.2** | *76.0* | *76.2* | *76.5*
| **Ours(3-cps)**| 321x321 | *74.9* | **76.4** | **76.3** | **76.6** |
| SupOnly | 513x513 | 54.1 | 60.7 | 67.7 | 71.9 |
| U<sup>2</sup>PL | 512x512 | - | 72.0 | 75.1 | 76.2 |
| PS-MT | 512x512 | - | 72.8 | 75.7 | 76.4 |
| **Ours(2-cps)**| 513x513 | **75.2** | *76.2* | *77.0* | *77.5* |
| **Ours(3-cps)**| 513x513 | *74.7* | **76.3** | **77.2** | **77.7** |

Results obtained by DeepLabv3+ with ResNet-101 and SegFormer-b3.

| Method | Resolution | 1/16 (662) | 1/8 (1323) | 1/4 (2646) |
| :-------------------------: | :-------: | :-------: | :-------: | :---------: |
| SupOnly | 321x321 |  67.5 | 70.4 | 73.7 |
| CAC | 321x321 | 72.4 | 74.6 | 76.3 |
| CTT* | 321x321 | 73.7 | 75.1 | - |
| ST++ | 321x321 | 74.5 | 76.3 | 76.6 |
| **Ours(2-cps)** | 321x321 |  **77.6** | **78.3** | **78.7**
| **Ours(3-cps)**| 321x321 |  *77.3* | *78.0* | *78.6* |
| SupOnly | 513x513 | 66.6 | 70.5 | 74.5|
| MT | 512x512 | 70.6 |  73.2  | 76.6 | 
| CCT | 512x512 | 67.9  | 73.0 |  76.2 | 
| GCT | 512x512 | 67.2  | 72.2 |  73.6 | 
| CPS | 512x512 | 74.5  | 76.4 |  77.7 | 
| 3-CPS | 512x512 | 75.8  | 78.0  | 79.0 | 
| CutMix | 512x512 | 72.6 |  72.7 |  74.3 | 
| DSBN‡ | 769x769 | -  |  74.1  | 77.8 | 
| ELN | 512x512 | -  | 75.1  | 76.6 | 
| PS-MT | 512x512 | 75.5  | 78.2 |  78.7 | 
| AEL | 513x513 | 77.2  | 77.6  | 78.1 | 
| U<sup>2</sup>PL | 513x513 |  74.4 | 77.6 | 78.7 |
| **Ours(2-cps)**| 513x513 | **77.9** | *78.7* | *79.0* |
| **Ours(3-cps)**| 513x513 | *77.6* | **79.0** | **80.0** |

### Cityscapes

Results are obtained by DeepLabv3+ with ResNet-50/101 and SegFormer-b2/b3 with resolution 769x769. Results of U<sup>2</sup>PL are from [UniMatch](https://github.com/LiheYoung/UniMatch).

| ResNet-50                      | 1/30     | 1/8     | 1/4       | ResNet-101 | 1/16       | 1/8          | 1/4        |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: | :---------: | :---------: |
| SupOnly   | 54.8  | 70.2  | 73.6    | SupOnly  | 66.8 | 72.5  | 76.4 |
| U<sup>2</sup>PL             | 59.8      | 73.0      | 76.3      | U<sup>2</sup>PL      | 74.9        | 76.5        | 78.5        |
| ST++   | 61.4 | 72.7 |73.8 | PS-MT | - | 76.9 | 77.6  |
| **Ours (2-cps)**         | *64.5*  | *76.3*  | *77.1*  | **Ours (2-cps)**  | *75.0*    | *77.3*    | **78.7**    |
| **Ours (3-cps)**         | **65.5**  | **76.5**  | **77.9**  | **Ours (3-cps)**  | **75.7**    | **77.4**    | *78.5*   |

### Training Logs
We release the training logs in [logs folder](logs/). 
Training logs on CItyscapes dataset can be found in [logs/cityscapes](logs/cityscapes/).
Training logs on VOC dataset can be found in [logs/voc](logs/voc).
### Checkpoints
We will release the checkpoints later.

## Getting Started

### Installation

```bash
conda create -n cotraining python=3.7
conda activate cotraining
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

We use implementation of SegFormer from mmsegmentation, so installation of mmcv and mmsegmentation libraries is needed

```bash
pip install openmim
mim install mmcv
pip install mmsegmentation
```

Proprosessing of DCT domain requires jpeg2dct and PyTurboJPEG libraries.

Before installing jpeg2dct, first install either [libjpeg](https://libjpeg.sourceforge.net/) or [libjpeg-turbo](https://libjpeg-turbo.org/) library. 

```bash
apt install libjpeg-turbo
pip install jpeg2dct PyTurboJPEG
```

For more details regarding the installation of jpeg2dct, we refer to [jpeg2dct](https://github.com/uber-research/jpeg2dct).
We also refer to [DCTNet](https://github.com/kaixu-zen/DCTNet) for more details of DCT transform.

### Pretrained Backbone:
We provide the pretrain as followed:

[ResNet-50](https://drive.google.com/file/d/1-kNZxhBfTnOc4V2yBaegvfhmVBWh1ByV/view?usp=drive_link) | [ResNet-101](https://drive.google.com/file/d/1-bybHiZGvnLaH0JF7qZ81uPa9rEW53s2/view?usp=drive_link) | [ResNet-50-dct](https://drive.google.com/file/d/1-l4TWQSJIw3gP6_hSZezj91_RfuZpbQN/view?usp=drive_link) | [ResNet-101-dct](https://drive.google.com/file/d/1-jI4fsGs3oCbt5Idh-aV2x9xsQJXIUUj/view?usp=drive_link)

```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    ├── resnet50_dct.pth
    └── resnet101_dct.pth
```

**Note**: The ResNet variants all use official weights and we have pretrained DCT ResNet with comparable performance as ResNet couterparts. More details can be found in our paper.
### Dataset:

- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```

## Usage

### Diverse Co-training

```bash
# use torch.distributed.launch
# To start training, the general format is as followed
sh <script> <num_gpu> <port> <data partition> <threshold>
# e.g. bash tools/voc/dist_train_cotraining_2cps.sh 4 29873 1_16 0.0

# we also provide a srun script for training on slurm cluster
# e.g. bash tools/voc/srun_train_cotraining_2cps.sh 4 29873 1_16 0.0
```

In order to run on different labeled data partitions or different datasets, please modify:

``config``, ``labeled_id_path``, ``unlabeled_id_path``, and ``save_path`` in the training shell script.

### Supervised Baseline

Modify the py file to ``supervised.py`` in the script, and double the ``batch_size`` in the configuration file if you use the same number of GPUs as semi-supervised setting (no need to change ``lr``). 

If you want to run supervised on DCT input domain, follow the above instructions to modify ``supervised_dct.py`` in the script, everything is the same except for the input domain is changed.

## Citation
If you find this project useful, please consider citing:
```bibtex
@InProceedings{Li_2023_ICCV,
    author    = {Li, Yijiang and Wang, Xinjiang and Yang, Lihe and Feng, Litong and Zhang, Wayne and Gao, Ying},
    title     = {Diverse Cotraining Makes Strong Semi-Supervised Segmentor},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16055-16067}
}
```
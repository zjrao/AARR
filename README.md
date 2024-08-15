# AARR

This is an implementation for paper "Keep and Learn: Attribute-Aware Representation Rectification for Generalized Zero-Shot Learning".

## Environment

* cuda=11.1

* python=3.8.18

* torch=1.8.0

* torchvision=0.9.0

* GPU: V100 32GB

## Data Preparation

1.Download datasets and put them in ‘./data'. We use three datasets [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) following the data split of [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).

2.Run the script preprocessing.py to preprocess the data. For example:

```
python preprocessing.py --dataset CUB --compression --device cuda:0
```

## Test

Download our trained models from [Google Drive](https://drive.google.com/drive/folders/1JE0N5UMC4BZqDjhNs0ptik84_8Gemdjf) and put them in './model'. Run test scripts. For example:

```
python Test_cub.py
```

## Train

1.Run pretrain scripts to train classifiers. For example:

```
python Pretrain_cub.py
```

2.Run our scripts to get final models. For example:

```
python AARR_cub.py
```

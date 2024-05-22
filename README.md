# KOM-Euph
This repo contains source code and data to reproduce the results in the research paper: A Unified Generative Framework for Bilingual Euphemism Detection and Identification.
![image](https://github.com/DHZ68/KOM-Euph/assets/32569555/b70670fb-9ff9-4827-a66d-5abdd2f107cd)

## Table of Contents

- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Data](#Data)
- [Code](#Code)

## Introduction

This project aims to create a unified generative framework to solve the problem of euphemism detection and euphemism identification in bilingual languages.

## Requirements

The code is based on Python 3.8.18 Please install the dependencies as below:  

You should first download `Pytorch` corresponding to the system Cuda version. In the experimental environment of this paper, we used `cuda 11.4`, so the installation code is as follows. You can also go to the [Pytorch](https://pytorch.org/) official website to download other corresponding versions.

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Then install other required packages.

```
pip install -r requirements.txt
```

Since we use different models of `mBart` and `mT5` models for experiments, you need to download the corresponding model parameters on Hugging Face.
- [mbart-large-50](https://huggingface.co/facebook/mbart-large-50)
- [mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25)
- [mT5-base](https://huggingface.co/google/mt5-base)
- [mT5-large](https://huggingface.co/google/mt5-large)

## Data

The `data` folder contains folders in `Chinese` and `English`. There are `train.json`, `dev.json`, and `test.json` files in each language folder.

## Code

`main.py` describes the four modules of `parameter setting`, `constructing dataset`, `training model,` and `testing model`, and the specific implementation is distributed in other files of `utils`.
![image](https://github.com/DHZ68/KOM-Euph/assets/32569555/a6211343-8e71-4906-b885-fef9bd610291)
You can use the terminal to execute the `main.py` file, as follows:

```
python main.py \
    --dataset English \
    --model_name facebook/mbart-large-cc25 \
    --num_epochs 20 \
    --batch_size 32 \
    --lr 1e-5 \
    --save_model False \
    --random_seed 1334 \
    --alpha 0.5 \
    --beta 0.3 \
    --gamma 0.2
```

You can also directly execute the `run.sh` script file to run the code.

```
chmod +x run.sh
```

```
./run.sh
```


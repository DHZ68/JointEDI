#!/bin/bash
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
#!/usr/bin/env bash

# L2 正则实验：只修改 --l2，其它参数固定
# 基准：l2 = 1e-6

echo "Running TIGER l2 regularization sweep on Grocery_and_Gourmet_Food ..."

# 实验 1：不加 L2 正则
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 0 --dataset Grocery_and_Gourmet_Food --gpu 0

# 实验 2：原始 L2 强度（baseline）
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --gpu 0

# 实验 3：更强的 L2 正则
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-4 --dataset Grocery_and_Gourmet_Food --gpu 0
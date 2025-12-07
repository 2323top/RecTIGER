# 只改 history_max 默认20，发现10最佳
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 10 --dataset Grocery_and_Gourmet_Food --gpu 0
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 30 --dataset Grocery_and_Gourmet_Food --gpu 0
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Grocery_and_Gourmet_Food --gpu 0
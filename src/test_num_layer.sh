# 只改 num_layers 默认为4，发现2更好
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --num_layers 2 --dataset Grocery_and_Gourmet_Food --gpu 0
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --num_layers 4 --dataset Grocery_and_Gourmet_Food --gpu 0
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --num_layers 6 --dataset Grocery_and_Gourmet_Food --gpu 0
#跑GRU4Rec在两个模型上的数据
python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset MovieLens_1M/ML_1MTOPK --gpu 0

python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 0

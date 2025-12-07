# 只改 beam_size 默认30，发现这个不影响结果
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --beam_size 10 --dataset Grocery_and_Gourmet_Food --gpu 0
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --beam_size 30 --dataset Grocery_and_Gourmet_Food --gpu 0
python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --beam_size 50 --dataset Grocery_and_Gourmet_Food --gpu 0
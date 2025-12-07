# python main.py --model_name TIGER --emb_size 64 --lr 5e-4 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --gpu 0
# Dev  After Training: (HR@5:0.4357,NDCG@5:0.3399,HR@10:0.5264,NDCG@10:0.3691,HR@20:0.6372,NDCG@20:0.3970,HR@50:0.8274,NDCG@50:0.4346)
# Test After Training: (HR@5:0.3961,NDCG@5:0.3011,HR@10:0.4842,NDCG@10:0.3294,HR@20:0.5956,NDCG@20:0.3575,HR@50:0.7946,NDCG@50:0.3968)

# python main.py --model_name TIGER --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --gpu 0
# 40轮的结果为：HR@5:0.4240,NDCG@5:0.3239
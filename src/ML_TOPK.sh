# python main.py --model_name TIGER --dataset MovieLens_1M/ML_1MTOPK --gpu 0 --epoch 100 --regenerate 1 --batch_size 256 --eval_batch_size 128 --lr 5e-4 --l2 1e-6
# Dev  After Training: (HR@5:0.1148,NDCG@5:0.0715,HR@10:0.1928,NDCG@10:0.0965,HR@20:0.3251,NDCG@20:0.1295,HR@50:0.6542,NDCG@50:0.1942)
# Test After Training: (HR@5:0.1308,NDCG@5:0.0801,HR@10:0.2241,NDCG@10:0.1100,HR@20:0.3678,NDCG@20:0.1460,HR@50:0.6729,NDCG@50:0.2061)


python main.py --model_name TIGER --dataset MovieLens_1M/ML_1MTOPK --gpu 0 --epoch 100 --batch_size 256 --eval_batch_size 128 --lr 1e-3 --l2 2e-6 --num_layers 2 --history_max 10
#没跑过
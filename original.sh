# cross entropy
# CUDA_VISIBLE_DEVICES=$1 python main_ce.py --batch_size 64 \
#   --learning_rate 0.8 \
#   --cosine --syncBN \

# #sup con learning pretraining
CUDA_VISIBLE_DEVICES=$1 python main_supcon.py --batch_size 64 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine

# #linear training
# CUDA_VISIBLE_DEVICES=$1 python main_linear.py --batch_size 32 \
#   --learning_rate 5 \
#   --ckpt /path/to/model.pth

# #SimCLR pretraining
# CUDA_VISIBLE_DEVICES=$1 python main_supcon.py --batch_size 64 \
#   --learning_rate 0.5 \
#   --temp 0.5 \
#   --cosine --syncBN \
#   --method SimCLR

# #linear trainig
# CUDA_VISIBLE_DEVICES=$1 python main_linear.py --batch_size 32 \
#   --learning_rate 1 \
#   --ckpt /path/to/model.pth
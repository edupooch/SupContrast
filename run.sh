# # Sup con loss pretraining:
# CUDA_VISIBLE_DEVICES=$1 python main_supcon.py --batch_size 128  --learning_rate 0.5 --temp 0.1 --cosine --dataset 'cmnist' --data_folder '/C/eduardo/datasets/CMNIST' --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)'

# # # linear
# CUDA_VISIBLE_DEVICES=$1 python main_linear.py \
#    --batch_size 128 \
#    --learning_rate 5 \
#    --dataset 'cmnist' \
#    --data_folder '/C/eduardo/datasets/CMNIST' \
#    --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)' \
#    --ckpt 'save/SupCon/cmnist_models/SimCLR_cmnist_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.5_trial_0_cosine/ckpt_epoch_1000.pth' | tee save/linear-unsup-pre-1000ep-500val$(date +%y-%m-%d-%H-%M).txt
#    #  --ckpt 'save/SupCon/cmnist_models/SupCon_cmnist_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_400.pth' 
#    # --ckpt 'save/SupCon/cmnist_models/SupCon_cmnist_resnet50_lr_0.5_decay_0.0001_bsz_64_temp_0.1_trial_0_cosine/ckpt_epoch_1000.pth'


# # CE trainig
# CUDA_VISIBLE_DEVICES=$1 python main_ce.py --batch_size 256 \
#    --learning_rate 0.8  \
#    --cosine --syncBN \
#    --dataset 'cmnist' \
#    --data_folder '/C/eduardo/datasets/CMNIST' \
#    --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)' | tee save/ce-aug-$(date +%y-%m-%d-%H-%M).txt


#simclr pretraining
# CUDA_VISIBLE_DEVICES=$1 python main_supcon.py --batch_size 128 \
#   --learning_rate 0.5 \
#   --temp 0.5 \
#   --cosine --syncBN \
#   --method SimCLR \
#   --dataset 'cmnist' \
#   --data_folder '/C/eduardo/datasets/CMNIST' \
#   --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)' 
#    | tee save/simclr-pre-$(date +%y-%m-%d-%H-%M).txt
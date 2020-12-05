# # Sup con loss pretraining:
# CUDA_VISIBLE_DEVICES=$1 python main_supcon.py --batch_size 128  --learning_rate 0.5 --temp 0.1 --cosine --dataset 'cmnist' --data_folder '/C/eduardo/datasets/CMNIST' --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)'

# # # linear
CUDA_VISIBLE_DEVICES=$1 python main_linear.py \
   --batch_size 128 \
   --learning_rate 5 \
   --dataset 'celeba' \
   --data_folder '/A/ngavenski/git/adversarial-bias/' \
   --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)' \
   --ckpt '/C/eduardo/projects/SupContrast/save/SupCon/celeba_models/SimCLR_celeba_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.5_trial_0_cosine/ckpt_epoch_50.pth' | tee save/celeba-linear-simclr-pre-500ep$(date +%y-%m-%d-%H-%M).txt


# # CE trainig
# CUDA_VISIBLE_DEVICES=$1 python main_ce.py --batch_size 256 \
#    --learning_rate 0.8  \
#    --cosine --syncBN \
#    --dataset 'celeba' \
#    --data_folder '/A/ngavenski/git/adversarial-bias/' \
#    --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)' | tee save/ce-aug-$(date +%y-%m-%d-%H-%M).txt


#simclr pretraining
# CUDA_VISIBLE_DEVICES=$1 python main_supcon.py --batch_size=128 \
#   --learning_rate 0.5 \
#   --temp 0.5 \
#   --cosine --syncBN \
#   --method SimCLR \
#   --dataset 'celeba' \
#   --data_folder '/A/ngavenski/git/adversarial-bias/' \
#   --mean '(0.5071, 0.4867, 0.4408)' --std '(0.2675, 0.2565, 0.2761)'  
  #| tee save/celeba-simclr-pre-$(date +%y-%m-%d-%H-%M).txt
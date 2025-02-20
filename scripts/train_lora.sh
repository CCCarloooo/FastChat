cd /home/mxd/archive/FastChat

<<<<<<< HEAD
export NCCL_P2P_DISABLE="1" 
export NCCL_IB_DISABLE="1"

deepspeed --include "localhost:0,1,2,3,4,5,6,7" --master_port 20701 fastchat/train/train_lora.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path /home/mxd/archive/hf_mirror/llama-7b-hf \
    --lora_r 8 \
    --lora_alpha 16 \
    --adam_beta2 0.99 \
    --init_interval 400 \
    --interval 400 \
    --save_flag True \
    --custom_save_interval 2 \
    --lora_target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' 'gate_proj' 'up_proj' 'down_proj' \
    --data_path /home/mxd/archive/hf_mirror/multitask_61k/all_traindata_new_shuffle.json \
    --output_dir ./checkpoints/rank8_0124_constant_1e-4_099_6ep \
    --num_train_epochs 3 \
    --bf16 True \
    --per_device_train_batch_size 3 \
=======
deepspeed --include "localhost:3,4,5" --master_port 29801 fastchat/train/train_lora.py \
    --deepspeed /mnt/data2/mxdi/archive/FastChat/scripts/zero2.json \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b \
    --lora_r 1 \
    --lora_alpha 2 \
    --data_path /mnt/data2/mxdi/archive/ift_prac/multitask/all_traindata_new_shuffle.json \
    --output_dir ./checkpoints/a40_0115_rank1_1e-4_0.999_3epoch \
    --num_train_epochs 3 \
    --bf16 True \
    --per_device_train_batch_size 8 \
>>>>>>> 915a41ed69b18b072dff956efb6f14273ad00024
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100  \
<<<<<<< HEAD
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
=======
    --save_strategy "no" \
    --save_steps 1 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --lr_scheduler_type 'constant' \
>>>>>>> 915a41ed69b18b072dff956efb6f14273ad00024
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False \
    --lazy_preprocess True

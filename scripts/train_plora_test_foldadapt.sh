cd /home/mxd/archive/FastChat

export NCCL_P2P_DISABLE="1" 
export NCCL_IB_DISABLE="1"

deepspeed --include "localhost:0,1,2,3,4,5,6,7" --master_port 47891 fastchat/train/train_plora_now.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path /home/mxd/archive/hf_mirror/llama-7b-hf \
    --lora_r 8 \
    --lora_alpha 16 \
    --init_interval 250 \
    --interval 250 \
    --plora_momentum True \
    --plora_momentum_ratio 0.1 \
    --resume_pth False \
    --resume_pth_path /home/mxd/archive/FastChat/checkpoints/plora_r8_0119_interval400_1e-4_099_constant_07lora_3save/7569-savemodel.pth \
    --save_flag True \
    --custom_save_interval 2 \
    --adam_beta2 0.99 \
    --lora_target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' 'gate_proj' 'up_proj' 'down_proj' \
    --data_path /home/mxd/archive/hf_mirror/multitask_61k/all_traindata_new_shuffle.json \
    --output_dir ./checkpoints/plora_r8_250_01_save2 \
    --num_train_epochs 3 \
    --bf16 True \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100  \
    --save_strategy "no" \
    --save_steps 1 \
    --save_total_limit 30 \
    --lr_scheduler_type "constant" \
    --learning_rate 1e-4 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False \
    --lazy_preprocess True

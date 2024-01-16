cd /mnt/data2/mxdi/archive/FastChat/

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
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100  \
    --save_strategy "no" \
    --save_steps 1 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --lr_scheduler_type 'constant' \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False \
    --lazy_preprocess True

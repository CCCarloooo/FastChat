cd /mnt/data2/mxdi/archive/FastChat/

deepspeed --include "localhost:0,1,2,3" --master_port=20101 fastchat/train/train_mem.py \
    --deepspeed /mnt/data2/mxdi/archive/FastChat/scripts/zero2.json \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b  \
    --data_path /mnt/data2/mxdi/archive/ift_prac/all_traindata_new_new.json \
    --bf16 True \
    --output_dir ./checkpoints/finetuned_1000_z3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --eval_steps 1500 \
    --save_strategy "no" \
    --save_steps 10 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --lazy_preprocess True
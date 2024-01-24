cd /mnt/data2/mxdi/archive/FastChat/

deepspeed --include "localhost:0,1,2,3,4,5" --master_port=20101 fastchat/train/train_mem.py \
    --deepspeed /mnt/data2/mxdi/archive/FastChat/scripts/zero2.json \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b  \
    --data_path /mnt/data2/mxdi/archive/ift_prac/all_traindata_new_new.json \
    --bf16 True \
    --output_dir ./checkpoints/a40_finetuned \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 1500 \
    --save_strategy "no" \
    --save_steps 10 \
    --save_total_limit 8 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --lazy_preprocess True
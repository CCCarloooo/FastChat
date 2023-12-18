cd /mnt/data2/mxdi/archive/FastChat/

deepspeed --include "localhost:0,1,2,3" --master_port=20101 fastchat/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b  \
    --data_path /mnt/data2/mxdi/archive/hf-mirror/SlimOrca \
    --bf16 True \
    --output_dir output_openllama3b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 1500 \
    --save_strategy "no" \
    --save_steps 1500 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
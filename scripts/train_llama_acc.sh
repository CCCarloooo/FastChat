cd /mnt/data2/mxdi/archive/FastChat/

accelerate launch --config_file /mnt/data2/mxdi/archive/acce_config/test.yaml \
    fastchat/train/train_mem.py \
        --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b  \
        --data_path /mnt/data2/mxdi/archive/ift_prac/all_sampled1000.json \
        --bf16 True \
        --output_dir ./checkpoints/finetuned_1000_z3 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 2 \
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
        --model_max_length 1280 \
        --gradient_checkpointing True \
        --lazy_preprocess True
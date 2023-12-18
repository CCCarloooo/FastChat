cd /mnt/data2/mxdi/archive/FastChat

deepspeed --include=localhost:2,3,4,5 --master_port=12345 fastchat/train/train_mem.py \
    --deepspeed /mnt/data2/mxdi/archive/LLaVA/scripts/zero3.json \
    --model_name_or_path /mnt/data2/mxdi/archive/models/Llama-2-7b-hf  \
    --data_path /mnt/data2/mxdi/archive/your_dataset.json \
    --bf16 True \
    --output_dir output_gpt2-xl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True

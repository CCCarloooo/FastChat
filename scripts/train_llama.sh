cd /mnt/data2/mxdi/archive/FastChat/
# 训练 全量 仅gsm8k 2e-5 0.999 0.04 4*1*6

deepspeed --include "localhost:0,1,2,3,4,5" --master_port=20101 fastchat/train/train_mem.py \
    --deepspeed /mnt/data2/mxdi/archive/FastChat/scripts/zero2.json \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b  \
    --data_path /mnt/data2/mxdi/archive/ift_prac/multitask/all_traindata_new_shuffle.json \
    --bf16 True \
<<<<<<< HEAD
    --output_dir ./checkpoints/a40_finetuned \
    --num_train_epochs 1 \
=======
    --output_dir ./checkpoints/finetuned_4epoch \
    --num_train_epochs 3 \
>>>>>>> 915a41ed69b18b072dff956efb6f14273ad00024
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 1500 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 8 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --lazy_preprocess True
cd /mnt/data2/mxdi/archive/FastChat/

deepspeed --include "localhost:0,1,2,3,4,5" --master_port 21941 fastchat/train/train_plora_test_08lora.py \
    --deepspeed /mnt/data2/mxdi/archive/FastChat/scripts/zero2.json \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b \
    --lora_r 1 \
    --lora_alpha 2 \
    --init_interval 400 \
    --interval 200 \
    --adam_beta2 0.999 \
    --data_path /mnt/data2/mxdi/archive/ift_prac/multitask/all_traindata_new_shuffle.json \
    --output_dir ./checkpoints/a40_plora_r1_0116_interval400-200_1e-4_0999_constant_07lora \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100  \
    --save_strategy "no" \
    --save_steps 1 \
    --save_total_limit 30 \
    --lr_scheduler_type "constant" \
    --learning_rate 1e-4 \
<<<<<<< HEAD
    --weight_decay 0. \
    --warmup_steps 40 \
    --lr_scheduler_type "constant_with_warmup" \
=======
>>>>>>> 915a41ed69b18b072dff956efb6f14273ad00024
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False \
    --lazy_preprocess True



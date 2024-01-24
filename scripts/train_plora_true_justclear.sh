cd /mnt/data2/mxdi/archive/FastChat/

# gsm8k rank1 plora 100 1e-4 0.999
# /mnt/data2/mxdi/archive/ift_prac/math_sys/math_train.json
# /mnt/data2/mxdi/archive/ift_prac/multitask/all_traindata_new_shuffle.json
deepspeed --include "localhost:0,1,2,3,4,5" --master_port 45891 fastchat/train/train_plora_test.py \
    --deepspeed /mnt/data2/mxdi/archive/FastChat/scripts/zero2.json \
    --model_name_or_path /mnt/data2/mxdi/archive/hf-mirror/llama-7b \
    --lora_r 1 \
    --lora_alpha 2 \
    --init_interval 300 \
    --interval 300 \
    --adam_beta2 0.999 \
    --data_path /mnt/data2/mxdi/archive/ift_prac/multitask/all_traindata_new_shuffle.json \
    --output_dir ./checkpoints/plora_multitask_r1_0112_qv_interval300_1e-4_0.999_enoughsave_scaling_constant \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100  \
    --save_strategy "no" \
    --save_steps 1000 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False \
    --lazy_preprocess True

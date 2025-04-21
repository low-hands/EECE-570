# 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model 'C:\Users\Ray\Desktop\eece570\practice\Qwen2VL-model\Qwen2-VL-2B\qwen\Qwen2-VL-2B-Instruct' \
    --train_type lora \
    --dataset 'D:\UBC\eece570\data\coco_final_train.json' \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --lazy_tokenize true \
    --target_modules "all-linear" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --learning_rate 5e-5 \
    --split_dataset_ratio 0.0 \
    --dataset_num_proc 1 \
    --truncation_strategy delete \
    --gradient_accumulation_steps 4\
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --save_steps 500 \
    --logging_steps 5 \
    --max_length 2048 \
    --lr_scheduler_type cosine \
    --output_dir 'D:\UBC\eece570\output' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --add_version \
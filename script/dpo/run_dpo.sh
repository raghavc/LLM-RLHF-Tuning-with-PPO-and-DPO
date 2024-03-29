
pretrained_model=alpaca_path
dataset_dir=/root/LLM-RLHF-Tuning/rm_data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
output_dir=dpo_lora_path


accelerate launch --config_file default_config.yaml run_dpo_with_peft.py \
    --model_type llama \
    --template "llama2_alpaca" \
    --model_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --split_ratio 0.05 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --seed 512 \
    --fp16 \
    --num_train_epochs 1 \
    --max_prompt_length 512 \
    --max_response_length 512 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 1 \
    --output_dir ${output_dir} \
    --logging_first_step True \
    --lora_rank 128 \
    --lora_alpha 32 \
    --lora_target ${lora_trainable} \
    --lora_dropout 0.05 \
    --report_to "wandb" \
    --dpo_beta 1.0 \
    --torch_dtype float16
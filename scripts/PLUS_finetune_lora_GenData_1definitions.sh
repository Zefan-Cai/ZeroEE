export CUDA_VISIBLE_DEVICES=3

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=32
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./open_instruct/ds_configs/stage3_no_offloading_accelerate.conf \
    ./open_instruct/open_instruct/finetune_GenData.py \
    --model_name_or_path /local1/zefan/models/Llama-2-7b-hf/ \
    --use_flash_attn \
    --use_lora \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --tokenizer_name /local1/zefan/models/Llama-2-7b-hf/ \
    --use_slow_tokenizer \
    --train_file /local1/zefan/data/generated_data/train_1definitions.json \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir /local1/zefan/output/Llama-2-7b-GenData-1definitions/ \
    --save_merged_lora_model \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path /local1/zefan/models/Llama-2-7b-hf/ \
    --lora_model_name_or_path  /local1/zefan/output/Llama-2-7b-GenData-1definitions/
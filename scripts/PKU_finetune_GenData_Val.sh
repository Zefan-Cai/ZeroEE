export CUDA_VISIBLE_DEVICES="1,2"
​
# Modify Arguments Here Before Training
MODEL_PATH="output/TULU-LLAMA2-7B"
OUTPUT_NAME="DEBUG"
TRAIN_FILE="example_val.jsonl"
# wp_pretrain_100K.jsonl
# TRAIN_FILE="data/processed/writing_prompts/wp_allcont_debug.jsonl"
VAL_FILE="example_val.jsonl"
TEST_FILE="example_val.jsonl"
REPORT_TAGS="CtrlGen"
#
ceildiv(){ echo $((($1+$2-1)/$2)); }
NUM_GPUS=$(ceildiv ${#CUDA_VISIBLE_DEVICES} 2)
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=4
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model ${MODEL_PATH} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
​
accelerate launch \
    --main_process_port 22455 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune_for_zefan.py \
    --model_name_or_path $MODEL_PATH \
    --use_flash_attn \
    --tokenizer_name $MODEL_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --val_file $TEST_FILE \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/$OUTPUT_NAME \
    --with_tracking \
    --report_to wandb \
    --report_name $OUTPUT_NAME \
    --report_tags $REPORT_TAGS \
    --eval_steps 2 \
    --checkpointing_steps epoch \
    --logging_steps 1
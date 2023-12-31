export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH="/home/models/Llama-2-7b-hf/"
FINETUNE_MODEL_PATH="/home/caizf/projects/ZeroEE/output/Llama2_GenData_1definitions_v2/checkpoint-5"


OUTPUT_NAME=Llama2_GenData_1definitions_v2/
TRAIN_FILE="/home/caizf/projects/ZeroEE/data/generated_data/train_1definitions_100.json"
VAL_FILE="/home/caizf/projects/ZeroEE/data/generated_data/val_1definitions.json"
TEST_FILE="/home/caizf/projects/ZeroEE/data/ace_v2/ACE_valid_GenerationStyle_trigger.json"
REPORT_TAGS="CtrlGen"

# ceildiv(){ echo $((($1+$2-1)/$2)); }
# NUM_GPUS=$(ceildiv ${#CUDA_VISIBLE_DEVICES} 2)
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=4
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model ${MODEL_PATH} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 22455 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./open_instruct/ds_configs/stage3_no_offloading_accelerate.conf \
    ./open_instruct/open_instruct/finetune_val.py \
    --model_name_or_path $FINETUNE_MODEL_PATH \
    --use_flash_attn \
    --tokenizer_name $MODEL_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --val_file $TEST_FILE \
    --test_file $VAL_FILE \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /home/caizf/projects/ZeroEE/output/${OUTPUT_NAME} \
    --with_tracking \
    --report_to wandb \
    --report_name $OUTPUT_NAME \
    --report_tags $REPORT_TAGS \
    --eval_steps 4 \
    --checkpointing_steps epoch \
    --logging_steps 1
export CUDA_VISIBLE_DEVICES="0,1,2"

MODEL_PATH="/local1/zefan/models/Llama-2-7b-hf/"

FINETUNE_MODEL="/local1/zefan/output/Llama-2-7b-geneva-20-96-2000/epoch_10"

OUTPUT_NAME=GenData_100_1definition_debug/
TRAIN_FILE=" /local1/zefan/data_event_number/geneva/GENEVA_train_negatives15_samples10_events40.json"
VAL_FILE="/local1/zefan/data/ace_v1/ACE_valid_v1_trigger.json"
TEST_FILE="/local1/zefan/data/ace_v1/ACE_test_v1_trigger.json"
METRICS_FILE="/local1/zefan/ZeroEE/open_instruct/compute_score_ee.py"
REPORT_TAGS="ZeroEE"

# ceildiv(){ echo $((($1+$2-1)/$2)); }
# NUM_GPUS=$(ceildiv ${#CUDA_VISIBLE_DEVICES} 2)
NUM_GPUS=3
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=48
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model ${MODEL_PATH} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 22455 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /local1/zefan/ZeroEE/open_instruct/ds_configs/stage3_no_offloading_accelerate.conf \
    /local1/zefan/ZeroEE/open_instruct/open_instruct/finetune_val.py \
    --model_name_or_path $MODEL_PATH \
    --use_flash_attn \
    --tokenizer_name $MODEL_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --test_file $TEST_FILE \
    --metrics_file ${METRICS_FILE} \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 20 \
    --output_dir /local1/zefan/output/${OUTPUT_NAME} \
    --with_tracking \
    --report_to wandb \
    --report_name $OUTPUT_NAME \
    --report_tags $REPORT_TAGS \
    --checkpointing_steps epoch \
    --logging_steps 1
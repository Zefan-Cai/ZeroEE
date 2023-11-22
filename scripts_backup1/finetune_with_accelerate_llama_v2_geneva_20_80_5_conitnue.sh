export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_SIZE=7b
NUM_GPUS=4
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training gt model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 29000 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./open_instruct/ds_configs/stage3_no_offloading_accelerate.conf \
    ./open_instruct/open_instruct/finetune.py \
    --model_name_or_path /local1/zefan/output/Llama-2-7b-geneva-20-77-5/epoch_5 \
    --use_flash_attn \
    --tokenizer_name /local1/zefan/models/Llama-2-7b-hf/ \
    --use_slow_tokenizer \
    --train_file /local1/zefan/data_event_number/geneva/GENEVA_train_negatives20_samples5_events77.json \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 20 \
    --output_dir /local1/zefan/output/Llama-2-${MODEL_SIZE}-geneva-20-77-5-continue/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --checkpointing_steps epoch
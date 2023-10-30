export CUDA_VISIBLE_DEVICES=2,3,5,6,7

MODEL_SIZE=7b
NUM_GPUS=5
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=40
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
    --model_name_or_path /home/models/Llama-2-7b-hf/ \
    --use_flash_attn \
    --tokenizer_name /home/models/Llama-2-7b-hf/ \
    --use_slow_tokenizer \
    --train_file /home/caizf/projects/ZeroEE/data_definition/GENEVA_train_negatives15_samples5_events77_definition6.json \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 30 \
    --output_dir /home/caizf/projects/ZeroEE/output/Llama-2-${MODEL_SIZE}-geneva-15-77-5-6/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --checkpointing_steps epoch
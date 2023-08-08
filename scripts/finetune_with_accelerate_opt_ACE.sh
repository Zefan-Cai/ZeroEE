export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=350m
NUM_GPUS=2
BATCH_SIZE_PER_GPU=64
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training opt model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 29005 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./open_instruct/ds_configs/stage3_no_offloading_accelerate.conf \
    ./open_instruct/open_instruct/finetune.py \
    --model_name_or_path /home/caizf/models/opt-1.3b/ \
    --use_flash_attn \
    --tokenizer_name /home/caizf/models/opt-350m/ \
    --use_slow_tokenizer \
    --train_file /home/caizf/projects/ZeroEE/data/ace/ACE_train_3.json \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 45 \
    --output_dir /home/caizf/projects/ZeroEE/output/opt_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
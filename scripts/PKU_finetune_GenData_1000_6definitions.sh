export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-22 23:03:35
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-11-23 14:08:54
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_finetune_GenData_1000_6definitions.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-22 23:03:35
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-11-22 23:15:39
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_finetune_GenData_1000_6definitions.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

MODEL_PATH="/home/models/Llama-2-7b-hf/"

OUTPUT_NAME=GenData_1000_6definition/
TRAIN_FILE=" /home/caizf/projects/ZeroEE/data/generated_data/train_1000_6definitions.json"
VAL_FILE="/home/caizf/projects/ZeroEE/data/generated_data/valid_100_1definitions.json"
TEST_FILE="/home/caizf/projects/ZeroEE/data/ace_v2/ACE_test_v2_trigger.json"
METRICS_FILE="/home/caizf/projects/ZeroEE/ZeroEE/open_instruct/compute_score_ee.py"
REPORT_TAGS="ZeroEE"

# ceildiv(){ echo $((($1+$2-1)/$2)); }
# NUM_GPUS=$(ceildiv ${#CUDA_VISIBLE_DEVICES} 2)
NUM_GPUS=6
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=96
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model ${MODEL_PATH} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 22455 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /home/caizf/projects/ZeroEE/ZeroEE/open_instruct/ds_configs/stage3_no_offloading_accelerate.conf \
    /home/caizf/projects/ZeroEE/ZeroEE/open_instruct/open_instruct/finetune_val.py \
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
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir /home/caizf/projects/ZeroEE/output/${OUTPUT_NAME} \
    --with_tracking \
    --report_to wandb \
    --report_name $OUTPUT_NAME \
    --report_tags $REPORT_TAGS \
    --checkpointing_steps epoch \
    --logging_steps 1
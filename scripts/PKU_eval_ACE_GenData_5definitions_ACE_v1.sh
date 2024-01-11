export CUDA_VISIBLE_DEVICES=$1
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2024-01-10 09:45:56
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-01-10 09:46:11
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_eval_ACE_GenData_5definitions_ACE_v1.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

MODEL_NAME="$2"

base_dir='/home/caizf/projects/ZeroEE/'

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir ${base_dir}data/ace_v2 \
    --valid_file ACE_valid_v2_inference.json \
    --test_file ACE_test_v2_inference.json \
    --save_dir ${base_dir}results/${MODEL_NAME} \
    --model ${base_dir}output/${MODEL_NAME} \
    --tokenizer ${base_dir}output/${MODEL_NAME} \
    --eval_batch_size 8


export CUDA_VISIBLE_DEVICES=4
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-09-19 00:24:22
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-09-19 00:24:35
 # @FilePath: /ZeroEE/ZeroEE/scripts/92_eval_ACE_test_Llama_v2_geneva_15_77_10.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-77-10-epoch10/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-77-10/epoch_10/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-77-10-epoch20/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-77-10/epoch_20/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-77-10-epoch29/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-77-10/epoch_29/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8
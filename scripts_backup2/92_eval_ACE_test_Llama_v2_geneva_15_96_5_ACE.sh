export CUDA_VISIBLE_DEVICES=2
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-09-19 00:17:53
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-09-19 00:24:18
 # @FilePath: /ZeroEE/ZeroEE/scripts/92_eval_ACE_test_Llama_v2_geneva_15_96_5_ACE.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-96-5-ACE-epoch10/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-5-96-ACE/epoch_10/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8


# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-96-5-ACE-epoch20/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-5-96-ACE/epoch_20/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8


# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-96-5-ACE-epoch29/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-5-96-ACE/epoch_29/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

export CUDA_VISIBLE_DEVICES=0
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-09-19 00:54:58
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-09-19 00:55:26
 # @FilePath: /ZeroEE/ZeroEE/scripts/eval_ACE_test_Llama_v2_geneva_15_77_5.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/test_Llama-2-7b-geneva-15-77-5-epoch10/ \
    --model /local1/zefan/output/Llama-2-7b-geneva-15-5-77/epoch_10/ \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/test_Llama-2-7b-geneva-15-77-5-epoch20/ \
    --model /local1/zefan/output/Llama-2-7b-geneva-15-5-77/epoch_20/ \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/test_Llama-2-7b-geneva-15-77-5-epoch29/ \
    --model /local1/zefan/output/Llama-2-7b-geneva-15-5-77/epoch_29/ \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8


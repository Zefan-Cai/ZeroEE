export CUDA_VISIBLE_DEVICES=1
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-25 14:11:51
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-01-31 15:54:10
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_eval_ACE_GenData1000_6definitions_ACE_v2.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###


base_dir=/home/caizf/projects/ZeroEE


python -m open_instruct.eval.ace.run_eval \
    --data_dir ${base_dir}/data/ace_v2 \
    --valid_file ACE_valid_v2_inference.json \
    --test_file ACE_test_v2_inference.json \
    --save_dir ${base_dir}/results/GenevaDatav2_Samples40_events96_1definition_epoch5_ACE_ACEDefinition/ \
    --model ${base_dir}/output/GenevaDatav2_Samples40_events96_1definition/epoch_5 \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8


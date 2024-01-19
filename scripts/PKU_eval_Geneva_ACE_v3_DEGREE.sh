export CUDA_VISIBLE_DEVICES=7
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-25 14:11:51
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-01-11 13:20:17
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_eval_ACE_GenData1000_6definitions_ACE_v2.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###


base_dir=/home/caizf/projects/ZeroEE/

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir ${base_dir}data/ace_v3 \
    --valid_file ACE_valid_v3_DEGREE_inference.json \
    --test_file ACE_test_v3_DEGREE_inference.json \
    --save_dir ${base_dir}results/Geneva_77_10sample_epoch25_ACEv3DEGREE/ \
    --model ${base_dir}output/Llama-2-7b-geneva-15-77-10/epoch_25 \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 16


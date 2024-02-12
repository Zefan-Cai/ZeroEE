export CUDA_VISIBLE_DEVICES="$1"
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2024-01-31 15:53:48
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-02-01 14:52:30
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_eval_GenevaDatav2_ACE_ACEDefinition.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
export Epoch="$2"

base_dir=/local1/zefan

python -m open_instruct.eval.ace.run_eval \
    --data_dir ${base_dir}/data/ace_v2 \
    --valid_file ACE_valid_v2_Degree_inference.json \
    --test_file ACE_test_v2_Degree_inference.json \
    --save_dir ${base_dir}/results/GENEVA_n10_s2000_e96_6d_v2_GenData_e200_10d_5s_v2_epoch${Epoch}_oldACE_DegreeDefi/ \
    --model ${base_dir}/output/GENEVA_n10_s2000_e96_6d_v2_GenData_e200_10d_5s_v2/epoch_${Epoch} \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 64

export CUDA_VISIBLE_DEVICES="$1"
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2024-01-31 15:53:48
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-02-01 14:52:30
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_eval_GenevaDatav2_ACE_ACEDefinition.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
export NumEvent="$2"
export NumDefinition="$3"
export Epoch="$4"
export Dataset="$5"

base_dir=/local1/zefan

python -m open_instruct.eval.ace.run_eval \
    --data_dir ${base_dir}/data/${Dataset}_v2_textEE \
    --valid_file ${Dataset}_valid_v2_inference.json \
    --test_file ${Dataset}_test_v2_inference.json \
    --save_dir ${base_dir}/results/GenevaDatav2_Samples2000_events${NumEvent}_${NumDefinition}definition_epoch${Epoch}_${Dataset}/ \
    --model ${base_dir}/output/GenevaDatav2_Samples2000_events${NumEvent}_${NumDefinition}definition/epoch_${Epoch} \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 64

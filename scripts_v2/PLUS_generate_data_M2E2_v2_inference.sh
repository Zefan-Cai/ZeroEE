export CUDA_VISIBLE_DEVICES=1
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2024-01-31 19:02:28
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-01-31 19:37:05
 # @FilePath: /ZeroEE/ZeroEE/scripts_v2/PKU_generate_data_CASIE_v2_inference.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

service=pku
template=v2

dataset_name='M2E2'
setting='inference'

definition_file=${dataset_name}_event_definition.json
ontology_file=${dataset_name}_ontology.json

base_dir='/local1/zefan'
data_info_dir='ZeroEE/data'
data_dir='processed_data'
output_dir='data/'${dataset_name}'_'${template}'_textEE'

mkdir ${base_dir}/${output_dir}

output_valid_filename=${dataset_name}'_valid_'${template}'_'${setting}'.json'
output_test_filename=${dataset_name}'_test_'${template}'_'${setting}'.json'


python ./generate_ACE_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --data_info_dir ${data_info_dir} \
    --output_dir ${output_dir} \
    --data_dir ${data_dir} \
    --output_valid_filename ${output_valid_filename} \
    --output_test_filename ${output_test_filename} \
    --definition_file ${definition_file} \
    --ontology_file ${ontology_file} \
    --dataset_name ${dataset_name} \
    --setting ${setting}

















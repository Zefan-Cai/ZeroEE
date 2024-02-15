export CUDA_VISIBLE_DEVICES=1
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2024-01-31 19:02:28
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-02-01 14:42:22
 # @FilePath: /ZeroEE/ZeroEE/scripts_v2/PKU_generate_data_CASIE_v2_inference.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

service=pku
template=v2

dataset_name='ACE'
setting='evaluation'

definition_file=${dataset_name}_event_definition_v0.json
ontology_file=${dataset_name}_ontology_vo.json

base_dir='/cpfs01/shared/public/caizefan.czf/ZeroEE'


data_info_dir='ZeroEE/data/'
output_dir='data/ace_v2'
definition_file='ACE_event_definition_v0.json'
ontology_file='ACE_ontology_v0.json'

# output_valid_filename='ACE_valid_v2_trigger.json'
# output_test_filename='ACE_test_v2_trigger.json'

# If add setting inference, use this name
output_valid_filename=ACE_valid_v2_${setting}.json
output_test_filename=ACE_test_v2_${setting}.json

python ./generate_rawACE_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --data_info_dir ${data_info_dir} \
    --output_dir ${output_dir} \
    --output_valid_filename ${output_valid_filename} \
    --output_test_filename ${output_test_filename} \
    --definition_file ${definition_file} \
    --ontology_file ${ontology_file} \
    --setting ${setting}
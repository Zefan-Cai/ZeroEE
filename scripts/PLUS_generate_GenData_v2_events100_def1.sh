
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-22 18:59:18
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-12-23 00:00:40
 # @FilePath: /ZeroEE/ZeroEE/scripts/PLUS_generate_GenData_v2_events100_def1.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
service=plus
template=v2

base_dir="/local1/zefan/"
generate_data_dir='ZeroEE/data/generated_data_fix_v4.json'
output_dir='data/generated_data'

output_train_filename='train_100_1definition.json'
output_valid_filename='valid_100_1definition.json'

num_definitions=1

train_parent_start=50
train_parent_end=100
valid_parent_start=0
valid_parent_end=50

python ./generate_train_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --generate_data_dir ${generate_data_dir} \
    --output_dir ${output_dir} \
    --output_train_filename ${output_train_filename} \
    --output_valid_filename ${output_valid_filename} \
    --num_definitions ${num_definitions} \
    --train_parent_start ${train_parent_start} \
    --train_parent_end ${train_parent_end} \
    --valid_parent_start ${valid_parent_start} \
    --valid_parent_end ${valid_parent_end}
    

















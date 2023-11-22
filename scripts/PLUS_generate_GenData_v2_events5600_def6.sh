
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-22 21:31:30
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-11-22 23:35:10
 # @FilePath: /ZeroEE/ZeroEE/scripts/PLUS_generate_GenData_v2_events6700_def6.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
service=pku
template=v2

base_dir="/local1/zefan/"
# base_dir="/home/caizf/projects/ZeroEE/"
generate_data_dir='ZeroEE/data/generated_data_fix.json'
output_dir='data/generated_data'

output_train_filename='train_5600_6definition.json'
output_valid_filename='valid_100_1definition.json'

num_definitions=5

train_parent_start=50
train_parent_end=1750
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
    

















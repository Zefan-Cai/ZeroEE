###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-12-23 12:13:54
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-01-08 16:03:57
 # @FilePath: /ZeroEE/ZeroEE/scripts/PLUS_generate_GenData_v3_events200_def5.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###

# The updated scripts, will use the specific amount of events
service=pku
template=v2

event_num=$1

base_dir='/cpfs01/user/caizefan/ZeroEE/'

generate_data_dir='ZeroEE/data/generated_data_fix_v5.json'
output_dir='data/generated_data'

output_train_filename=train_${event_num}_10definitions_v2.json
output_valid_filename='valid_100_1definitions_v2.json'

num_negative_sample=10
num_negative_inOntology=3
num_definitions=10

train_parent_start=50
train_parent_end=1000000
valid_parent_start=0
valid_parent_end=50

random_seed=42

python ./generate_train_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --generate_data_dir ${generate_data_dir} \
    --output_dir ${output_dir} \
    --output_train_filename ${output_train_filename} \
    --output_valid_filename ${output_valid_filename} \
    --num_negative_sample ${num_negative_sample} \
    --num_negative_inOntology ${num_negative_inOntology} \
    --num_definitions ${num_definitions} \
    --random_seed ${random_seed} \
    --train_parent_start ${train_parent_start} \
    --train_parent_end ${train_parent_end} \
    --valid_parent_start ${valid_parent_start} \
    --valid_parent_end ${valid_parent_end} \
    --train_events $event_num


















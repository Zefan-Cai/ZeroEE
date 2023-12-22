
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-22 22:59:23
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-12-22 22:54:07
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_generate_GenData_v2_events200_def6.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
service=pku
template=v2

base_dir='/home/caizf/projects/ZeroEE/'
generate_data_dir='ZeroEE/data/generated_data_fix.json'
output_dir='data/generated_data'

output_train_filename='train_200_5definitions.json'
output_valid_filename='valid_100_1definitions.json'

num_negative_sample=15
num_negative_inOntology=3

train_parent_start=50
train_parent_end=150
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
    --num_negative ${num_negative} \
    --num_negative_inOntology ${num_negative_inOntology} \
    --random_seed ${random_seed} \
    --train_parent_start ${train_parent_start} \
    --train_parent_end ${train_parent_end} \
    --valid_parent_start ${valid_parent_start} \
    --valid_parent_end ${valid_parent_end}


















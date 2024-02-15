
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2024-02-01 14:48:59
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-02-02 15:02:18
 # @FilePath: /ZeroEE/ZeroEE/scripts_v2/PKU_generate_GenevaData_v2.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

service=pku
template=v2

base_dir='/cpfs01/shared/public/caizefan.czf/ZeroEE'
event_definition_dir='GENEVA_event_definition_6.json'
output_dir='geneva_train_v2_data'

least_sampleNum_perEvent='0'

num_negative_sample=10
num_definitions=1
random_seed=42




python ./generate_GENEVA_train_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --event_definition_dir ${event_definition_dir} \
    --output_dir ${output_dir} \
    --least_sampleNum_perEvent ${least_sampleNum_perEvent} \
    --num_negative_sample ${num_negative_sample} \
    --num_definitions ${num_definitions} \
    --random_seed ${random_seed} \
    --add_definition True
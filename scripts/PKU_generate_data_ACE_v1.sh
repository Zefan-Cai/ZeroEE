
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-22 18:59:18
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-11-22 20:49:09
 # @FilePath: /ZeroEE/ZeroEE/scripts/PLUS_generate_data_ACE_v1.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
service=pku
template=v1

base_dir='/home/caizf/projects/ZeroEE/'
data_info_dir='/ZeroEE/data/'
output_dir='/data/ace_v1'

output_valid_filename='ACE_valid_v1_trigger.json'
output_test_filename='ACE_test_v1_trigger.json'

python ./generate_ACE_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --data_info_dir ${data_info_dir} \
    --output_dir ${output_dir} \
    --output_valid_filename ${output_valid_filename} \
    --output_test_filename ${output_test_filename}

















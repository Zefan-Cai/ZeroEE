
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-25 14:26:46
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2023-11-28 18:12:35
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_generate_data_ACE_v2_inference.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
service=pku
template=v2

base_dir='/home/caizf/projects/ZeroEE'
data_info_dir='ZeroEE/data'
output_dir='data/ace_v2'

output_valid_filename='ACE_valid_v2_inference.json'
output_test_filename='ACE_test_v2_inference.json'

setting='inference'

python ./generate_ACE_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --data_info_dir ${data_info_dir} \
    --output_dir ${output_dir} \
    --output_valid_filename ${output_valid_filename} \
    --output_test_filename ${output_test_filename} \
    --setting ${setting}


















service=pku
template=v2

base_dir='/home/caizf/projects/ZeroEE'
data_info_dir='ZeroEE/data'
output_dir='data/ace_v2'

output_valid_filename='ACE_valid_v2_trigger.json'
output_test_filename='ACE_test_v2_trigger.json'

python ./generate_ACE_data.py \
    --service ${service} \
    --template_version ${template} \
    --base_dir ${base_dir} \
    --data_info_dir ${data_info_dir} \
    --output_dir ${output_dir} \
    --output_valid_filename ${output_valid_filename} \
    --output_test_filename ${output_test_filename}

















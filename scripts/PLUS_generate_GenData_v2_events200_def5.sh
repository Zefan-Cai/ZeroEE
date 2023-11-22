
service=plus
template=v2

base_dir="/local1/zefan/"
generate_data_dir='ZeroEE/data/generated_data_fix.json'
output_dir='data/generated_data'

output_train_filename='train_200_5definition.json'
output_valid_filename='valid_100_1definition.json'

num_definitions=5

train_parent_start=50
train_parent_end=150
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
    

















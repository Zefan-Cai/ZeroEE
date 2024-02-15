

service=pku
template=v2

base_dir="/local1/zefan"
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
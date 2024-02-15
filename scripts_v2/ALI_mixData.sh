

base_dir='/cpfs01/shared/public/caizefan.czf/ZeroEE/'
geneva_dir='data/geneva_train_v2_data/GENEVA_train_negatives10_samples2000_events96_v2_6definition.jsonl'
generated_dir='data/generated_data/train_200_10definitions_v2.json'
output_dir='data/mix_data/GENEVA_n10_s2000_e96_6d_v2_GenData_e200_10d_5s_v2.json'

python ./mix_data.py \
    --geneva_dir ${base_dir}${geneva_dir} \
    --generated_dir ${base_dir}${generated_dir} \
    --output_dir ${base_dir}${output_dir}


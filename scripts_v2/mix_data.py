import json

base_dir = '/local1/zefan'

GenData_data = []
with open('/local1/zefan/data/generated_data/train_200_10definitions_v2.json', 'r') as fp:
    for line in fp.readlines():
        GenData_data.append(json.loads(line))
        

Geneva_data = []
with open('/local1/zefan/data/geneva_train_v2_data/GENEVA_train_negatives10_samples2000_events96_v2_6definition.jsonl', 'r') as fp:
    for line in fp.readlines():
        Geneva_data.append(json.loads(line))

final_data = GenData_data + Geneva_data

with open('/local1/zefan/data/mix_v2/GENEVA_n10_s2000_e96_6d_v2_GenData_e200_10d_5s_v2.json', 'w') as fp:
    for line in final_data:
        fp.write(json.dumps(line) + '\n')
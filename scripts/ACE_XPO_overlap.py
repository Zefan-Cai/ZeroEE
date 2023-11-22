import json

with open('../data/ACE_ontology.json', 'r') as fp:
    ace_ontology = json.load(fp)

acc_ontology_full_list = []

for key in ace_ontology.keys():
    acc_ontology_full_list.append(key)
    for item in ace_ontology[key]:
        acc_ontology_full_list.append(item)

with open('../data/generated_data.json', 'r') as fp:
    generated_data = json.load(fp)


deep_delete_list = []
for parent in generated_data.keys():
    delete_list = []
    if generated_data[parent]["data"] == {}:
        deep_delete_list.append(parent)
    for event in generated_data[parent]["data"].keys():
        # print(f"debug event {event}")
        if generated_data[parent]["data"][event] == {}:
            delete_list.append(event)
        elif generated_data[parent]["data"][event]["samples"] == []:
            delete_list.append(event)

    for event in delete_list:
        del generated_data[parent]["data"][event]
        if event in generated_data[parent]["sons"]:
            generated_data[parent]["sons"].remove(event)
        generated_data[parent]["events"].remove(event)

for parent in generated_data.keys():
    delete_list = []
    if generated_data[parent]["data"] == {}:
        deep_delete_list.append(parent)

for event in deep_delete_list:
    del generated_data[event]

delete_list = list(set(delete_list))
deep_delete_list = list(set(deep_delete_list))





print(f"debug len parents {len(generated_data)}")

generated_data_full_list = []
for key in generated_data.keys():
    for item in generated_data[key]['events']:
        generated_data_full_list.append(item)

print(f"debug len(generated_data_full_list) {len(generated_data_full_list)}")

delete_list = []

for key in generated_data.keys():
    for item in generated_data[key]['events']:
        if item in acc_ontology_full_list:
            delete_list.append(key)

delete_list = list(set(delete_list))

for delete_item in delete_list:
    del generated_data[delete_item]

generated_data_full_list = []
for key in generated_data.keys():
    for item in generated_data[key]['events']:
        generated_data_full_list.append(item)

print(f"debug len(generated_data_full_list) {len(generated_data_full_list)}")



with open('../data/generated_data_fix.json', 'w') as fp:
    json.dump(generated_data, fp, indent=4)
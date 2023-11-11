import os
import json
import copy
import random
from tqdm import tqdm

output_dir = "/local1/zefan/data/ace_v2_withEvent"







with open('/local1/zefan/ZeroEE/data/ACE_ontology.json', 'r') as fp:
    ACE_ontology = json.load(fp)

# ACE

## ACE Event Definition

with open('/local1/zefan/ZeroEE/data/ACE_event_definition.json', 'r') as fp:
    event_type2definition = json.load(fp)

event_list = list(event_type2definition.keys())


## ACE val data

ACE_valid_data = []

with open('/local1/zefan/ZeroEE/oneie_ace05_en_event/val.json', 'r') as fp:
    for line in fp.readlines():
        ACE_valid_data.append(json.loads(line))
        
valid_data = []

for index in range(len(ACE_valid_data)):
    
    event_type2trigger = {}
    sample_data_list = []
    
    for event_index in range(len(ACE_valid_data[index]["event"])):
        event_type = ACE_valid_data[index]["event"][event_index]["type"]
        trigger = ACE_valid_data[index]["event"][event_index]["text"]
        
        
        if event_type not in event_type2trigger.keys():
            event_type2trigger[event_type] = []
        event_type2trigger[event_type].append(trigger)

    for event_type in event_type2definition.keys():
        event_definition = event_type2definition[event_type]
        
        for parent_event in ACE_ontology.keys():
            if event_type == parent_event: break
            if event_type in ACE_ontology[parent_event]:break
        
        sons = ACE_ontology[parent_event]
        text_sons = ", ".join(sons)
        
        if event_type in event_type2trigger.keys():
            trigger = event_type2trigger[event_type]
            trigger = " and ".join(trigger)
        else:
            trigger = "<trigger>"
        
        sample = ACE_valid_data[index]["text"]
        
            
        sample_data_list.append({
            "Event definition": event_definition,
            "Event type": event_type,
            "prompt": f"SENTENCE: {sample} \n EVENT TYPE: {event_type}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
            "trigger": trigger
            })
    valid_data.append(sample_data_list)
    

with open(os.path.join(output_dir, f'ACE_valid_GenerationStyle.json'), 'w') as fp:
    for d in valid_data:
       json.dump(d, fp)
       fp.write('\n')







selected_valid_data = random.choices(valid_data, k=3)
       
with open(os.path.join(output_dir, f'ACE_valid_GenerationStyle_clean.json'), 'w') as fp:
    for d in selected_valid_data:
       json.dump(d, fp)
       fp.write('\n')



























## ACE val data

ACE_valid_data = []

with open('/local1/zefan/ZeroEE/oneie_ace05_en_event/val.json', 'r') as fp:
    for line in fp.readlines():
        ACE_valid_data.append(json.loads(line))
        
valid_data = []

for index in range(len(ACE_valid_data)):
    
    event_type2trigger = {}
    sample_data_list = []
    
    for event_index in range(len(ACE_valid_data[index]["event"])):
        event_type = ACE_valid_data[index]["event"][event_index]["type"]
        trigger = ACE_valid_data[index]["event"][event_index]["text"]
        
        
        if event_type not in event_type2trigger.keys():
            event_type2trigger[event_type] = []
        event_type2trigger[event_type].append(trigger)

    for event_type in event_type2definition.keys():
        event_definition = event_type2definition[event_type]
        
        for parent_event in ACE_ontology.keys():
            if event_type == parent_event: break
            if event_type in ACE_ontology[parent_event]:break
        
        sons = ACE_ontology[parent_event]
        text_sons = ", ".join(sons)
        
        if event_type in event_type2trigger.keys():
            trigger = event_type2trigger[event_type]
            trigger = " and ".join(trigger)
        else:
            trigger = "<trigger>"
        
        sample = ACE_valid_data[index]["text"]
        
        event_type_id = event_list.index(event_type)
            
        sample_data_list.append({
            # "Event definition": event_definition,
            "event_type": event_type_id,
            "prompt": f"SENTENCE: {sample} \n EVENT TYPE: {event_type}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
            "completion": f"Event trigger is ",
            "trigger": trigger
            })
    valid_data.append(sample_data_list)
    

with open(os.path.join(output_dir, f'ACE_valid_v2_trigger.json'), 'w') as fp:
    for index_i in range(len(valid_data)):
        for index_j in range(len(valid_data[index_i])):
            json.dump(valid_data[index_i][index_j], fp)
            fp.write('\n')








































       
## ACE test data

ACE_test_data = []

with open('/local1/zefan/ZeroEE/oneie_ace05_en_event/test.json', 'r') as fp:
    for line in fp.readlines():
        ACE_test_data.append(json.loads(line))


test_data = []

for index in range(len(ACE_test_data)):

    if ACE_test_data[index]["event"] == []: pass
    else:
        event_type2trigger = {}
        sample_data_list = []
        
        for event_index in range(len(ACE_test_data[index]["event"])):
            event_type = ACE_test_data[index]["event"][event_index]["type"]
            trigger = ACE_test_data[index]["event"][event_index]["text"]
            
            if event_type not in event_type2trigger.keys():
                event_type2trigger[event_type] = []
            event_type2trigger[event_type].append(trigger)

        for parent_event in ACE_ontology.keys():
            if event_type == parent_event: break
            if event_type in ACE_ontology[parent_event]:break
        
        sons = ACE_ontology[parent_event]
        text_sons = ", ".join(sons)

        for event_type in event_type2definition.keys():
            event_definition = event_type2definition[event_type]
            if event_type in event_type2trigger.keys():
                trigger = event_type2trigger[event_type]
                trigger = " and ".join(trigger)
            else:
                trigger = "<trigger>"
            
            sample = ACE_valid_data[index]["text"]
            
            event_type_id = event_list.index(event_type)
            
            sample_data_list.append({
                # "Event definition": event_definition,
                "event_type": event_type_id,       
                "prompt": f"SENTENCE: {sample} \n EVENT TYPE: {event_type}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                "completion": f"Event trigger is ",
                "trigger": trigger
                })
        test_data.append(sample_data_list)
    
with open(os.path.join(output_dir, f'ACE_test_v2_trigger.json'), 'w') as fp:
    for index_i in range(len(test_data)):
        for index_j in range(len(test_data[index_i])):
            json.dump(test_data[index_i][index_j], fp)
            fp.write('\n')
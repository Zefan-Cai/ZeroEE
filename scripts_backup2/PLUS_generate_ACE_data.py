import os
import json

output_dir = "/local1/zefan/data/ace"

# ACE

## ACE Event Definition

with open('./data/ACE_event_definition.json', 'r') as fp:
    event_type2definition = json.load(fp)

## ACE val data

ACE_valid_data = []

with open('oneie_ace05_en_event/val.json', 'r') as fp:
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
        if event_type in event_type2trigger.keys():
            trigger = event_type2trigger[event_type]
        else:
            trigger = "<trigger>"
        sample_data_list.append({
            "Event definition": event_definition,
            "Event type": event_type,       
            "prompt": "{} \n {} \n So what is the trigger?".format(ACE_valid_data[index]["text"], event_definition),
            "trigger": trigger
            })
    valid_data.append(sample_data_list)
    

with open(os.path.join(output_dir, f'ACE_valid_GenerationStyle.json'), 'w') as fp:
    for d in valid_data:
       json.dump(d, fp)
       fp.write('\n')
       
       
## ACE test data

ACE_test_data = []

with open('oneie_ace05_en_event/test.json', 'r') as fp:
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

        for event_type in event_type2definition.keys():
            event_definition = event_type2definition[event_type]
            if event_type in event_type2trigger.keys():
                trigger = event_type2trigger[event_type]
            else:
                trigger = "<trigger>"
            sample_data_list.append({
                "Event definition": event_definition,
                "Event type": event_type,       
                "prompt": "{} \n {} \n So what is the trigger?".format(ACE_valid_data[index]["text"], event_definition),
                "trigger": trigger
                })
        test_data.append(sample_data_list)
    
with open(os.path.join(output_dir, f'ACE_test_GenerationStyle.json'), 'w') as fp:
    for d in test_data:
       json.dump(d, fp)
       fp.write('\n')
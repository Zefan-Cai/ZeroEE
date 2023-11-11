
output_dir = "/local1/zefan/data"

import os
import random
import json

# GENEVA

event_type2definition = {}

with open('GENEVA-main/meta_data/event_ontology.json', 'r') as fp:
    raw_event_ontology= json.load(fp)


for key in raw_event_ontology.keys():
    event_type2definition[key] = raw_event_ontology[key]['description']

with open('./data/GENEVA_event_definition.json', 'w') as fp:
    json.dump(event_type2definition, fp)


event_list = list(event_type2definition.keys())

for num_event in [60, 100, 140, 179]:
    
    available_events = event_list[:num_event]
    avalibale_event_type2definition = {}
    for key, item in event_type2definition.items():
        if key in available_events:
            avalibale_event_type2definition[key] = item

    ## Geneva Train data

    GENEVA_training_data = []

    with open('GENEVA-main/data/train.json', 'r') as fp:
        for line in fp.readlines():
            GENEVA_training_data.append(json.loads(line))


    n_negative = 15

    positive_train_data = []
    negative_train_data = []
    train_data = []

    overlap_events = ['Telling', 'Arrest', 'Social_event', 'Come_together', 'Supply', 'Getting', 'Commerce_sell', 'Giving', 'Commerce_buy', 'Earnings_and_losses', 'Receiving', 'Exchange', 'Commerce_pay', 'Death', 'Bodily_harm', 'Protest', 'Communication', 'Traveling', 'Attack']


    for index in range(len(GENEVA_training_data)):
        
        GENEVA_training_data[index]["sentence"] = GENEVA_training_data[index]["sentence"].replace("\'\'", "")
        
        event_type2trigger = {}
        for event_index in range(len(GENEVA_training_data[index]["event_mentions"])):
            
            event_type = GENEVA_training_data[index]["event_mentions"][event_index]["event_type"]

            if event_type in available_events and event_type not in overlap_events:
                trigger = GENEVA_training_data[index]["event_mentions"][event_index]["trigger"]["text"]
                
                if event_type not in event_type2trigger.keys():
                    event_type2trigger[event_type] = []
                event_type2trigger[event_type].append(trigger)

        for event_type in event_type2trigger.keys():
            event_definition = avalibale_event_type2definition[event_type]
            positive_train_data.append({
                "Event definition": event_definition,
                "Event type": event_type,       
                "prompt": "{} \n {} \n So what is the trigger?".format(GENEVA_training_data[index]["sentence"], event_definition),
                "completion": "Event trigger is {}".format(" and ".join(event_type2trigger[event_type]))
                })
        
        if event_type2trigger != {}:
            
            available_evet_types = list(set(available_events) - set(event_type2trigger.keys()))
            selected_event_type = random.sample(available_evet_types, n_negative)
            
            for event_type in selected_event_type:
                event_definition = avalibale_event_type2definition[event_type]
                negative_train_data.append({
                    "Event definition": event_definition,
                    "Event type": event_type,       
                    "prompt": "{} \n {} \n So what is the trigger?".format(GENEVA_training_data[index]["sentence"], event_definition),
                    "completion": "Event trigger is <trigger>"
                    })

    train_data = positive_train_data + negative_train_data

    with open(os.path.join(output_dir, 'geneva', f'GENEVA_train_{str(n_negative)}_{str(num_event)}.json'), 'w') as fp:
        for line in train_data:
            json.dump(line, fp)
            fp.write('\n')

# ACE

## ACE Event Definition

with open('./data/ACE_event_definition_DEGREE.json', 'r') as fp:
    event_type2definition = json.load(fp)

## ACE train data


n_negative = 15
for n_negative in [0, 8, 15]:
    ACE_train_data = []

    with open('oneie_ace05_en_event/train.json', 'r') as fp:
        for line in fp.readlines():
            ACE_train_data.append(json.loads(line))

    positive_train_data = []
    negative_train_data = []
    train_data = []

    for index in range(len(ACE_train_data)):
        if len(ACE_train_data[index]["event"]) == 0: pass
        else:
            event_type2trigger = {}
            for event_index in range(len(ACE_train_data[index]["event"])):
                event_type = ACE_train_data[index]["event"][event_index]["type"]
                trigger = ACE_train_data[index]["event"][event_index]["text"]
                
                if event_type not in event_type2trigger.keys():
                    event_type2trigger[event_type] = []
                event_type2trigger[event_type].append(trigger)

            for event_type in event_type2trigger.keys():
                event_definition = event_type2definition[event_type]
                positive_train_data.append({
                    "Event definition": event_definition,
                    "Event type": event_type,       
                    "prompt": "{} \n {} \n So what is the trigger?".format(ACE_train_data[index]["text"], event_definition),
                    "completion": "Event trigger is {}".format(" and ".join(event_type2trigger[event_type]))
                    })
            
            available_evet_types = list(set(event_type2definition.keys()) - set(event_type2trigger.keys()))
            selected_event_type = random.sample(available_evet_types, n_negative)
            
            for event_type in selected_event_type:
                event_definition = event_type2definition[event_type]
                negative_train_data.append({
                    "Event definition": event_definition,
                    "Event type": event_type,       
                    "prompt": "{} \n {} \n So what is the trigger?".format(ACE_train_data[index]["text"], event_definition),
                    "completion": "Event trigger is <trigger>"
                    })

    train_data = positive_train_data + negative_train_data

    with open(os.path.join(output_dir, 'ace', f'ACE_train_{str(n_negative)}.json'), 'w') as fp:
        for d in train_data:
            json.dump(d, fp)
            fp.write('\n')


## ACE valid data

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
    

with open(os.path.join(output_dir, 'ace', f'ACE_valid.json'), 'w') as fp:
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
                "prompt": "{} \n {} \n So what is the trigger?".format(ACE_test_data[index]["text"], event_definition),
                "trigger": trigger
                })
        test_data.append(sample_data_list)
    
with open(os.path.join(output_dir, 'ace', f'ACE_test.json'), 'w') as fp:
    for d in test_data:
       json.dump(d, fp)
       fp.write('\n')
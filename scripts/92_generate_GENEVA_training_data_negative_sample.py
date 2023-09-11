
# output_dir = "/local1/zefan/data_event_number"
output_dir = "/home/caizf/projects/ZeroEE/data_event_number"


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

with open('./data/times2events.json', 'r') as fp:
    times2events = json.load(fp)
    
overlap_events = ['Telling', 'Arrest', 'Social_event', 'Come_together', 'Supply', 'Getting', 'Commerce_sell', 'Giving', 'Commerce_buy', 'Earnings_and_losses', 'Receiving', 'Exchange', 'Commerce_pay', 'Death', 'Bodily_harm', 'Protest', 'Communication', 'Traveling', 'Attack']

# event_list = list(event_type2definition.keys())

for num_sample in [0, 1, 5, 10, 20, 40, 80]:
    
    event_list = times2events[str(num_sample)]
    
    if num_sample == 0: num_sample = 2000
    if num_sample == 1: num_sample = 5
    
    number_of_events = len(event_list)
    
    available_events = event_list
    avalibale_event_type2definition = {}
    for key, item in event_type2definition.items():
        if key in available_events:
            avalibale_event_type2definition[key] = item

    ## Geneva Train data

    GENEVA_training_data = []

    with open('GENEVA-main/data/train.json', 'r') as fp:
        for line in fp.readlines():
            GENEVA_training_data.append(json.loads(line))

    for n_negative in [5, 10, 15, 20]:
        
        if n_negative > number_of_events: continue

        positive_train_data = []
        negative_train_data = []
        train_data = []

        event2times = {}

        for index in range(len(GENEVA_training_data)):
            
            GENEVA_training_data[index]["sentence"] = GENEVA_training_data[index]["sentence"].replace("\'\'", "")
            
            event_type2trigger = {}
            for event_index in range(len(GENEVA_training_data[index]["event_mentions"])):
                
                event_type = GENEVA_training_data[index]["event_mentions"][event_index]["event_type"]

                if event_type in available_events and event_type not in overlap_events:
                    trigger = GENEVA_training_data[index]["event_mentions"][event_index]["trigger"]["text"]

                    if event_type not in event2times.keys():
                        event2times[event_type] = 0
                    
                    if event2times[event_type] <= num_sample:
                        if event_type not in event_type2trigger.keys():
                            event_type2trigger[event_type] = []
                        event_type2trigger[event_type].append(trigger)
                        event2times[event_type] += 1


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
                if len(available_evet_types) > n_negative:
                    selected_event_type = random.sample(available_evet_types, n_negative)
                else:
                    selected_event_type = random.sample(available_evet_types, len(available_evet_types))
                
                for event_type in selected_event_type:
                    event_definition = avalibale_event_type2definition[event_type]
                    negative_train_data.append({
                        "Event definition": event_definition,
                        "Event type": event_type,       
                        "prompt": "{} \n {} \n So what is the trigger?".format(GENEVA_training_data[index]["sentence"], event_definition),
                        "completion": "Event trigger is <trigger>"
                        })

        train_data = positive_train_data + negative_train_data

        with open(os.path.join(output_dir, 'geneva', f'GENEVA_train_negatives{str(n_negative)}_samples{str(num_sample)}_events{str(number_of_events)}.json'), 'w') as fp:
            for line in train_data:
                json.dump(line, fp)
                fp.write('\n')

import os
import json
import copy
import random
from tqdm import tqdm

output_dir = "/local1/zefan/data"

with open('../data/generated_data.json', 'r') as fp:
    data = json.load(fp)

positive_train_data = []
negative_train_data = []

error_num = 0

for parent_event in data.keys():
    
    sons = data[parent_event]["sons"]
    events = data[parent_event]["events"]
    
    text_sons = ", ".join(sons)
    

    for event in events:
        
        negative_events = copy.deepcopy(events)
        negative_events.remove(event)
        
        if "name" in data[parent_event]["data"][event].keys():
            event_name = data[parent_event]["data"][event]["name"]
            event_definition = data[parent_event]["data"][event]["definition"]
            triggers = data[parent_event]["data"][event]["triggers"]
            samples = data[parent_event]["data"][event]["samples"]

            diverse_definitions = copy.deepcopy(data[parent_event]["data"][event]["rewrite_definitions"])
            diverse_definitions = diverse_definitions[:4]
            diverse_definitions.append(event_definition)
            
            
            for definition in diverse_definitions:
                for sample in samples:
                    
                    sentence = sample["sentence"]
                    trigger = sample["trigger"]
                    
                    selected_trigger = random.choice(triggers)
                
                    positive_train_data.append({
                        "Event definition": definition,
                        "Event type": event,
                        "Event name": event_name,    
                        "Event triggers": triggers,
                        "trigger": trigger,
                        "selected_trigger": selected_trigger,
                        "sentence": sentence,
                        "parent": parent_event,
                        "events": events,
                        "sons": sons,
                        "prompt": f"{sentence} \n The event is: {event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n Possibile triggers include: {selected_trigger}. \n So what is the trigger?",
                        "completion": f"Event trigger is {trigger}."
                        })

                for negative_event in negative_events:
                    
                    if "name" in data[parent_event]["data"][negative_event].keys():
                        negative_event_name = data[parent_event]["data"][negative_event]["name"]
                        negative_event_definition = data[parent_event]["data"][negative_event]["definition"]
                        negative_triggers = data[parent_event]["data"][negative_event]["triggers"]
                        negative_samples = data[parent_event]["data"][negative_event]["samples"]
                        
                        for negative_sample in negative_samples:
                            
                            negative_sentence = negative_sample["sentence"]
                            negative_trigger = negative_sample["trigger"]
                            
                            negative_selected_trigger = random.choice(negative_triggers)
                        
                            negative_train_data.append({
                                "Event definition": definition,
                                "Event type": event,
                                "Event name": event_name,    
                                "Event triggers": triggers,
                                "trigger": "<trigger>",
                                "selected_trigger": selected_trigger,
                                "sentence": negative_sentence,
                                "parent": parent_event,
                                "events": events,
                                "sons": sons,
                                "prompt": f"{negative_sentence} \n The event is: {event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n Possibile triggers include: {selected_trigger}. \n So what is the trigger?",
                                "completion": f"Event trigger is <trigger>."
                                })
        else:
            error_num += 1

print(f"debug len positive_train_data {str(len(positive_train_data))}")
print(f"debug len negative_train_data {str(len(negative_train_data))}")
print(f"debug error_num {str(error_num)}")

train_data =  positive_train_data + negative_train_data

with open(os.path.join(output_dir, 'generated_data', 'data_5definitions.json'), 'w') as fp:
    for line in tqdm(train_data):
        json.dump(line, fp)
        fp.write('\n')









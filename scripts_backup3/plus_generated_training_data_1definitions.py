import os
import json
import copy
import random
from tqdm import tqdm


base_dir = "/local1/zefan/"

output_dir = os.path.join(base_dir, "data")


with open(os.path.join(base_dir, 'ZeroEE', "data", 'ACE_ontology.json'), 'r') as fp:
    data = json.load(fp)



print(f"debug number of parents: {len(list(data.keys()))}")


all_parent_list = list(data.keys())

count_all_event = 0
for parent_event in all_parent_list:
    count_all_event += len(data[parent_event]["events"])

print(f"debug count_all_event {count_all_event}")








val_parent_list = list(data.keys())[:50]

count_val_event = 0
for parent_event in val_parent_list:
    count_val_event += len(data[parent_event]["events"])

print(f"debug count_val_event {count_val_event}")






train_parent_list_100 = list(data.keys())[50:100]

count_train_100_event = 0
for parent_event in train_parent_list_100:
    count_train_100_event += len(data[parent_event]["events"])

print(f"debug count_train_100_event {count_train_100_event}")





train_parent_list_400 = list(data.keys())[50:184]

count_train_400_event = 0
for parent_event in train_parent_list_400:
    count_train_400_event += len(data[parent_event]["events"])

print(f"debug count_train_400_event {count_train_400_event}")














val_event_list = []
val_event_name_definitions = {}
for parent_event in val_parent_list:
    
    val_event_name_definitions[parent_event] = {}
    
    for event in data[parent_event]["data"].keys():
        
        val_event_list.append(event)
        
        val_event_name_definitions[parent_event][event] = {
            "definition": data[parent_event]["data"][event]["definition"],
            "name": data[parent_event]["data"][event]["name"]
            }












positive_train_data = []
negative_train_data = []
error_num = 0


for parent_event in train_parent_list_100:
    
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
            
            
            # for definition in diverse_definitions:
            for sample in samples:
                
                sentence = sample["sentence"]
                trigger = sample["trigger"]
                
                selected_trigger = random.choice(triggers)
            
                positive_train_data.append({
                    # "Event definition": definition,
                    # "Event type": event,
                    # "Event name": event_name,    
                    # "Event triggers": triggers,
                    # "trigger": trigger,
                    # "selected_trigger": selected_trigger,
                    # "sentence": sentence,
                    # "parent": parent_event,
                    # "events": events,
                    # "sons": sons,
                    "prompt": f"SENTENCE: {sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
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
                            # "Event definition": definition,
                            # "Event type": event,
                            # "Event name": event_name,    
                            # "Event triggers": triggers,
                            # "trigger": "<trigger>",
                            # "selected_trigger": selected_trigger,
                            # "sentence": negative_sentence,
                            # "parent": parent_event,
                            # "events": events,
                            # "sons": sons,
                            "prompt": f"SENTENCE: {negative_sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                            "completion": f"Event trigger is <trigger>."
                            })
        else:
            error_num += 1

print(f"debug len positive_train_data {str(len(positive_train_data))}")
print(f"debug len negative_train_data {str(len(negative_train_data))}")
print(f"debug error_num {str(error_num)}")

train_data =  positive_train_data + negative_train_data

with open(os.path.join(output_dir, 'generated_data', 'train_1definitions_100.json'), 'w') as fp:
    for line in tqdm(train_data):
        json.dump(line, fp)
        fp.write('\n')


























positive_train_data = []
negative_train_data = []

for parent_event in train_parent_list_400:
    
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
            
            
            # for definition in diverse_definitions:
            for sample in samples:
                
                sentence = sample["sentence"]
                trigger = sample["trigger"]
                
                selected_trigger = random.choice(triggers)
            
                positive_train_data.append({
                    # "Event definition": definition,
                    # "Event type": event,
                    # "Event name": event_name,    
                    # "Event triggers": triggers,
                    # "trigger": trigger,
                    # "selected_trigger": selected_trigger,
                    # "sentence": sentence,
                    # "parent": parent_event,
                    # "events": events,
                    # "sons": sons,
                    "prompt": f"SENTENCE: {sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
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
                            # "Event definition": definition,
                            # "Event type": event,
                            # "Event name": event_name,    
                            # "Event triggers": triggers,
                            # "trigger": "<trigger>",
                            # "selected_trigger": selected_trigger,
                            # "sentence": negative_sentence,
                            # "parent": parent_event,
                            # "events": events,
                            # "sons": sons,
                            "prompt": f"SENTENCE: {negative_sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                            "completion": f"Event trigger is <trigger>."
                            })
        else:
            error_num += 1

print(f"debug len positive_train_data {str(len(positive_train_data))}")
print(f"debug len negative_train_data {str(len(negative_train_data))}")
print(f"debug error_num {str(error_num)}")

train_data =  positive_train_data + negative_train_data

with open(os.path.join(output_dir, 'generated_data', 'train_1definitions_400.json'), 'w') as fp:
    for line in tqdm(train_data):
        json.dump(line, fp)
        fp.write('\n')















































positive_val_data = []
negative_val_data = []

data_index = 0

for parent_event in val_parent_list:
    
    sons = data[parent_event]["sons"]
    events = data[parent_event]["events"]
    
    text_sons = ", ".join(sons)
    

    for event in events:
        
        negative_events = copy.deepcopy(events)
        negative_events.remove(event)
        
        if "name" in data[parent_event]["data"][event].keys():

            # event_name = data[parent_event]["data"][event]["name"]
            event_definition = data[parent_event]["data"][event]["definition"]
            # triggers = data[parent_event]["data"][event]["triggers"]
            # samples = data[parent_event]["data"][event]["samples"]

            # diverse_definitions = copy.deepcopy(data[parent_event]["data"][event]["rewrite_definitions"])
            # diverse_definitions = diverse_definitions[:4]
            # diverse_definitions.append(event_definition)
            
            event_id = val_event_list.index(event)
            
            # for definition in diverse_definitions:
            for sample in samples:
                
                # sentence = sample["sentence"]
                # trigger = sample["trigger"]
                
                # selected_trigger = random.choice(triggers)
            
                positive_val_data.append({
                    # "Event definition": definition,
                    # "Event type": event,
                    # "Event name": event_name,    
                    # "Event triggers": triggers,
                    # "trigger": trigger,
                    # "selected_trigger": selected_trigger,
                    # "sentence": sentence,
                    # "parent": parent_event,
                    # "events": events,
                    # "sons": sons,
                    "data_id": data_index,
                    "event_type": event_id,     
                    "prompt": f"SENTENCE: {sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                    "completion": f"Event trigger is ",
                    "trigger": f"{trigger}"
                    })

                for negative_parent_event in val_event_name_definitions.keys():
                    
                    # negative_parent_event_name = val_event_name_definitions[negative_parent_event][negative_parent_event]["name"]
                    
                    
                    
                    son_events = list(val_event_name_definitions[negative_parent_event].keys())
                    son_events.remove(negative_parent_event)
                    text_sons = ", ".join(son_events)
                                    
                    for negative_event in val_event_name_definitions[negative_parent_event].keys():
                        
                        if negative_event == event: continue
                    
                        # negative_event_name = data[parent_event]["data"][negative_event]["name"]
                        negative_event_definition = val_event_name_definitions[negative_parent_event][negative_event]["definition"]

                        # negative_triggers = data[parent_event]["data"][negative_event]["triggers"]
                        # negative_samples = data[parent_event]["data"][negative_event]["samples"]
                        
                        # for negative_sample in negative_samples:
                            
                            # negative_sentence = negative_sample["sentence"]
                            # negative_trigger = negative_sample["trigger"]
                            
                            # negative_selected_trigger = random.choice(negative_triggers)
                    
                        event_id = val_event_list.index(negative_event)
                    
                        negative_val_data.append({
                            # "Event definition": definition,
                            # "Event type": event,
                            # "Event name": event_name,    
                            # "Event triggers": triggers,
                            # "trigger": "<trigger>",
                            # "selected_trigger": selected_trigger,
                            # "sentence": negative_sentence,
                            # "parent": parent_event,
                            # "events": events,
                            # "sons": sons,
                            "data_id": data_index,
                            "event_type": event_id,   
                            "prompt": f"SENTENCE: {negative_event} \n EVENT TYPE: {negative_event_name}. \n DEFINITION: {negative_event_definition} \n PARENT: {negative_parent_event}, SON: {text_sons}. \n So what is the trigger?",
                            "completion": f"Event trigger is ",
                            "trigger":  "<trigger>"
                            })
                        
                data_index += 1
        else:
            error_num += 1

print(f"debug len positive_train_data {str(len(positive_val_data))}")
print(f"debug len negative_train_data {str(len(negative_val_data))}")
print(f"debug error_num {str(error_num)}")

val_data =  positive_val_data + negative_val_data

with open(os.path.join(output_dir, 'generated_data', 'val_1definitions_100.json'), 'w') as fp:
    for line in tqdm(val_data):
        json.dump(line, fp)
        fp.write('\n')

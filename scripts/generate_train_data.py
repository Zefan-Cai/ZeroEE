import os
import copy
import json
from tqdm import tqdm
import argparse


class Data():

    def __init__(self, args):
        
        self.args = args
        
        # Load Data
        with open(os.path.join(self.args.base_dir, self.args.generate_data_dir), 'r') as fp:
            self.data = json.load(fp)

        # num parents
        print(f"debug number of parents: {len(list(self.data.keys()))}")
        all_parent_list = list(self.data.keys())

        # num events
        count_all_event = 0
        for parent_event in all_parent_list:
            count_all_event += len(self.data[parent_event]["events"])
        print(f"debug count_all_event {count_all_event}")





        self.train_parent_list = list(self.data.keys())[self.args.train_parent_start:self.args.train_parent_end]

        count_train_event = 0
        for parent_event in self.train_parent_list:
            count_train_event += len(self.data[parent_event]["events"])

        print(f"debug count_train_event {count_train_event}")
        print(f"debug count_train_parent {self.args.train_parent_end - self.args.train_parent_start}")




        self.valid_parent_list = list(self.data.keys())[self.args.valid_parent_start:self.args.valid_parent_end]

        count_valid_event = 0
        for parent_event in self.valid_parent_list:
            count_valid_event += len(self.data[parent_event]["events"])

        print(f"debug count_valid_event {count_valid_event}")
        print(f"debug count_valid_parent {self.args.valid_parent_end - self.args.valid_parent_start}")





        self.valid_event_list = []
        self.valid_event_name_definitions = {}
        for parent_event in self.valid_parent_list:
            
            self.valid_event_name_definitions[parent_event] = {}
            
            for event in self.data[parent_event]["data"].keys():
                
                self.valid_event_list.append(event)
                
                self.valid_event_name_definitions[parent_event][event] = {
                    "definition": self.data[parent_event]["data"][event]["definition"],
                    "name": self.data[parent_event]["data"][event]["name"]
                    }




        self.get_train_data()
        self.get_valid_data()
    
    
    
    
    
    def get_train_data(self):
            
        positive_train_data = []
        negative_train_data = []
        error_num = 0

        for parent_event in self.train_parent_list:
            
            sons = self.data[parent_event]["sons"]
            events = self.data[parent_event]["events"]
            
            # For ontology information in the input
            text_sons = ", ".join(sons)

            for event in events:
                
                # Negative Sample
                negative_events = copy.deepcopy(events)
                if event in negative_events:
                    negative_events.remove(event)
            
                event_name = self.data[parent_event]["data"][event]["name"]
                event_definition = self.data[parent_event]["data"][event]["definition"]
                # triggers = self.data[parent_event]["data"][event]["triggers"]
                samples = self.data[parent_event]["data"][event]["samples"]

                diverse_definitions = copy.deepcopy(self.data[parent_event]["data"][event]["rewrite_definitions"])
                diverse_definitions = diverse_definitions[:self.args.num_definitions]
                diverse_definitions.append(event_definition)
                
                
                for definition in diverse_definitions:
                    for sample in samples:
                        
                        sentence = sample["sentence"]
                        trigger = sample["trigger"]
                        
                        # selected_trigger = random.choice(triggers)
                                
                        if self.args.template_version == "v1":
                            prompt = f"{sentence} \n The event is: {event}. \n The event definition is: {event_definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                        elif self.args.template_version == "v2":
                            prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                        else:
                            raise Exception("THe template version should be v1 or v2")
                    
                        if type(prompt) == list: prompt = prompt[0]
                        
                        positive_train_data.append({
                            "prompt": prompt[0],
                            "completion": f"Event trigger is {trigger}."
                            })

                    # Negative Sampling based on negative events
                    for negative_event in negative_events:
                        
                        if "name" in self.data[parent_event]["data"][negative_event].keys():
                            # negative_event_name = self.data[parent_event]["data"][negative_event]["name"]
                            # negative_event_definition = self.data[parent_event]["data"][negative_event]["definition"]
                            # negative_triggers = self.data[parent_event]["data"][negative_event]["triggers"]
                            negative_samples = self.data[parent_event]["data"][negative_event]["samples"]
                            
                            for negative_sample in negative_samples:
                                
                                negative_sentence = negative_sample["sentence"]
                                # negative_trigger = negative_sample["trigger"]
                                # negative_selected_trigger = random.choice(negative_triggers)
                                        
                                
                                        
                                if self.args.template_version == "v1":
                                    prompt = f"{negative_sentence} \n The event is: {event}. \n The event definition is: {event_definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                                elif self.args.template_version == "v2":
                                    prompt = f"SENTENCE: {negative_sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                                else:
                                    raise Exception("THe template version should be v1 or v2")
                                    
                                    
                                negative_train_data.append({
                                    "prompt": prompt[0],
                                    "completion": f"Event trigger is <trigger>."
                                    })

        print(f"debug len positive_train_data {str(len(positive_train_data))}")
        print(f"debug len negative_train_data {str(len(negative_train_data))}")
        print(f"debug error_num {str(error_num)}")

        train_data =  positive_train_data + negative_train_data

        with open(os.path.join(self.args.base_dir, self.args.output_dir, self.args.output_train_filename), 'w') as fp:
            for line in tqdm(train_data):
                json.dump(line, fp)
                fp.write('\n')

    
    
    
    
    
    
    
    
    
    
    def get_valid_data(self):
        
        positive_valid_data = []
        negative_valid_data = []

        data_index = 0

        for parent_event in self.valid_parent_list:
            
            sons = self.data[parent_event]["sons"]
            events = self.data[parent_event]["events"]
            
            # For ontology information in the input
            text_sons = ", ".join(sons)
            

            for event in events:
                
                # For negative sampling
                negative_events = copy.deepcopy(events)
                negative_events.remove(event)
                

                event_definition = self.data[parent_event]["data"][event]["definition"]
                samples = self.data[parent_event]["data"][event]["samples"]
                
                # for evaluation
                event_id = self.valid_event_list.index(event)
                
                for sample in samples:
                    sentence = sample["sentence"]
                    trigger = sample["trigger"]
                
                    if self.args.template_version == "v1":
                        prompt = f"{sentence} \n The event is: {event}. \n The event definition is: {event_definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                    elif self.args.template_version == "v2":
                        prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                    else:
                        raise Exception("THe template version should be v1 or v2")

                    if type(prompt) == list: prompt = prompt[0]
                    
                    positive_valid_data.append({
                        "data_id": data_index,
                        "event_type": event_id,     
                        "prompt": prompt[0],
                        "completion": f"Event trigger is ",
                        "trigger": f"{trigger}"
                        })

                    # Negative Sampling
                    for negative_parent_event in self.valid_event_name_definitions.keys():
                        
                        # For ontoligy information
                        son_events = list(self.valid_event_name_definitions[negative_parent_event].keys())
                        son_events.remove(negative_parent_event)
                        text_sons = ", ".join(son_events)
                        
                        # Enumerate all negative event definitions
                        # Negative Sampling based on negative definitions
                        for negative_event in self.valid_event_name_definitions[negative_parent_event].keys():
                            
                            # Avoid using the definition for the current sampole as negative definition
                            if negative_event == event: continue
                            
                            negative_event_definition = self.valid_event_name_definitions[negative_parent_event][negative_event]["definition"]
                            event_id = self.valid_event_list.index(negative_event)
                            
                            if self.args.template_version == "v1":
                                prompt = f"{sentence} \n The event is: {negative_event}. \n The event definition is: {negative_event_definition} \n The parent event is {negative_parent_event}, son events include {text_sons}. \n So what is the trigger?",
                            elif self.args.template_version == "v2":
                                prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {negative_event}. \n DEFINITION: {negative_event_definition} \n PARENT: {negative_parent_event}, SON: {text_sons}. \n So what is the trigger?",
                            else:
                                raise Exception("THe template version should be v1 or v2")
                            
                            if type(prompt) == list: prompt = prompt[0]
                        
                            negative_valid_data.append({
                                "data_id": data_index,
                                "event_type": event_id,   
                                "prompt": prompt[0],
                                "completion": f"Event trigger is ",
                                "trigger":  "<trigger>"
                                })
                    
                    # Each sample stands for one data index
                    data_index += 1

        print(f"debug len positive_train_data {str(len(positive_valid_data))}")
        print(f"debug len negative_train_data {str(len(negative_valid_data))}")

        valid_data =  positive_valid_data + negative_valid_data

        with open(os.path.join(self.args.base_dir, self.args.output_dir, self.args.output_valid_filename), 'w') as fp:
            for line in tqdm(valid_data):
                json.dump(line, fp)
                fp.write('\n')






























def main():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--service', default='plus', type=str, help='plus or pku')
    parser.add_argument('--template_version', default='v2', type=str, help='v1 or v2')
    
    parser.add_argument('--base_dir', default="/local1/zefan/", type=str, help=' plus: /local1/zefan/   pku: /home/caizf/projects/ZeroEE/ ')
    parser.add_argument('--generate_data_dir', default='ZeroEE/data/generated_data.json', type=str, help='dir to generated data')
    parser.add_argument('--output_dir', default='generated_data', type=str, help='dir to generated data')
    
    parser.add_argument('--output_train_filename', default='train_100_1definition.json', type=str, help='train filename')
    parser.add_argument('--output_valid_filename', default='valid_100_1definition.json', type=str, help='valid filename')
    
    parser.add_argument('--num_definitions', default=1, type=int, help='')

    parser.add_argument('--train_parent_start', default=50, type=int, help='')
    parser.add_argument('--train_parent_end', default=100, type=int, help='')
    parser.add_argument('--valid_parent_start', default=0, type=int, help='')
    parser.add_argument('--valid_parent_end', default=50, type=int, help='')

    args = parser.parse_args()
    
    args.num_definitions = args.num_definitions - 1

    data = Data(args)




if __name__ == "__main__":
    main()
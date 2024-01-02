import os
import json
import argparse
from tqdm import tqdm




class Data():
    def __init__(self, args):
        self.args = args
        
        with open(os.path.join(self.args.base_dir, self.args.data_info_dir, 'ACE_ontology.json'), 'r') as fp:
            self.ACE_ontology = json.load(fp)

        ## ACE Event Definition
        with open(os.path.join(self.args.base_dir, self.args.data_info_dir, 'ACE_event_definition.json'), 'r') as fp:
            self.event_type2definition = json.load(fp)
        self.event_list = list(self.event_type2definition.keys())

        self.get_valid_data()
        self.get_test_data()

    def get_valid_data(self):

        ## ACE val data
        ACE_valid_data = []

        with open(os.path.join(self.args.base_dir, 'ZeroEE', 'oneie_ace05_en_event', 'val.json'), 'r') as fp:
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

            for event_type in self.event_type2definition.keys():
                event_definition = self.event_type2definition[event_type]
                
                for parent_event in self.ACE_ontology.keys():
                    if event_type == parent_event: break
                    if event_type in self.ACE_ontology[parent_event]:break
                
                sons = self.ACE_ontology[parent_event]
                text_sons = ", ".join(sons)
                
                if event_type in event_type2trigger.keys():
                    trigger = event_type2trigger[event_type]
                    if self.args.setting == "inference":
                        pass
                    elif self.args.setting == "evaluation":
                        trigger = " and ".join(trigger)
                else:
                    trigger = "<trigger>"
                
                sample = ACE_valid_data[index]["text"]
                
                event_type_id = self.event_list.index(event_type)
                    
                if self.args.template_version == "v1":
                    prompt = f"{sample} \n The event is: {event_type}. \n The event definition is: {event_definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                elif self.args.template_version == "v2":
                    prompt = f"SENTENCE: {sample} \n EVENT TYPE: {event_type}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                else:
                    raise Exception("THe template version should be v1 or v2")
                    
                if type(prompt) == list: prompt = prompt[0]

                if self.args.setting == "evaluation":
                    sample_data_list.append({
                        "data_id": index,
                        "event_type": event_type_id,
                        "prompt": prompt[0],
                        "completion": f"Event trigger is ",
                        "trigger": trigger
                        })
                elif self.args.setting == "inference":
                    sample_data_list.append({
                        "data_id": index,
                        "event_type": event_type_id,
                        "prompt": prompt[0],
                        "completion": f"Event trigger is ",
                        "trigger": trigger,
                        "Event definition": event_definition,
                        "Event type": event_type
                        })

            valid_data.append(sample_data_list)
            

        with open(os.path.join(self.args.base_dir, self.args.output_dir, self.args.output_valid_filename), 'w') as fp:
            if self.args.setting == "evaluation":
                for index_i in range(len(valid_data)):
                    for index_j in range(len(valid_data[index_i])):
                        json.dump(valid_data[index_i][index_j], fp)
                        fp.write('\n')
            elif self.args.setting == "inference":
                for index_i in range(len(valid_data)):
                    
                    
                    json.dump(valid_data[index_i], fp)
                    fp.write('\n')

    def get_test_data(self):

        ## ACE test data

        ACE_test_data = []

        with open(os.path.join(self.args.base_dir, 'ZeroEE', 'oneie_ace05_en_event', 'test.json'), 'r') as fp:
            for line in fp.readlines():
                ACE_test_data.append(json.loads(line))


        test_data = []

        for index in range(len(ACE_test_data)):

            event_type2trigger = {}
            sample_data_list = []

            if ACE_test_data[index]["event"] == []: pass
            else:
                for event_index in range(len(ACE_test_data[index]["event"])):
                    event_type = ACE_test_data[index]["event"][event_index]["type"]
                    trigger = ACE_test_data[index]["event"][event_index]["text"]

                    if event_type not in event_type2trigger.keys():
                        event_type2trigger[event_type] = []
                    event_type2trigger[event_type].append(trigger)



            for event_type in self.event_type2definition.keys():
                event_definition = self.event_type2definition[event_type]
                
                for parent_event in self.ACE_ontology.keys():
                    if event_type == parent_event: break
                    if event_type in self.ACE_ontology[parent_event]:break
                
                sons = self.ACE_ontology[parent_event]
                text_sons = ", ".join(sons)
                
                if event_type in event_type2trigger.keys():
                    trigger = event_type2trigger[event_type]
                    if self.args.setting == "inference":
                        pass
                    elif self.args.setting == "evaluation":
                        trigger = " and ".join(trigger)
                else:
                    trigger = "<trigger>"



                sample = ACE_test_data[index]["text"]
                
                event_type_id = self.event_list.index(event_type)
                
                if self.args.template_version == "v1":
                    prompt = f"{sample} \n The event is: {event_type}. \n The event definition is: {event_definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                elif self.args.template_version == "v2":
                    prompt = f"SENTENCE: {sample} \n EVENT TYPE: {event_type}. \n DEFINITION: {event_definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                else:
                    raise Exception("THe template version should be v1 or v2")
                    
                if type(prompt) == list: prompt = prompt[0]

                if self.args.setting == "evaluation":
                    sample_data_list.append({
                        "data_id": index,
                        "event_type": event_type_id,
                        "prompt": prompt[0],
                        "completion": f"Event trigger is ",
                        "trigger": trigger
                        })
                elif self.args.setting == "inference":
                    sample_data_list.append({
                        "data_id": index,
                        "event_type": event_type_id,
                        "prompt": prompt[0],
                        "completion": f"Event trigger is ",
                        "trigger": trigger,
                        "Event definition": event_definition,
                        "Event type": event_type
                        })
            test_data.append(sample_data_list)

        with open(os.path.join(self.args.base_dir, self.args.output_dir, self.args.output_test_filename), 'w') as fp:
            if self.args.setting=="evaluation":
                for index_i in range(len(test_data)):
                    for index_j in range(len(test_data[index_i])):
                        json.dump(test_data[index_i][index_j], fp)
                        fp.write('\n')
            elif self.args.setting=="inference":
                for index_i in range(len(test_data)):
                    json.dump(test_data[index_i], fp)
                    fp.write('\n')

















def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--service', default='plus', type=str, help='plus or pku')
    parser.add_argument('--template_version', default='v2', type=str, help='v1 or v2')
    parser.add_argument('--base_dir', default='/local1/zefan/', type=str, help=' plus: /local1/zefan/   pku: /home/caizf/projects/ZeroEE/ ')
    parser.add_argument('--data_info_dir', default='/ZeroEE/data/', type=str, help='dir to generated data')
    parser.add_argument('--output_dir', default='/data/ace_v2', type=str, help='dir to generated data')
    parser.add_argument('--output_test_filename', default='ACE_test_v2_trigger.json', type=str, help='train filename')
    parser.add_argument('--output_valid_filename', default='ACE_valid_v2_trigger.json', type=str, help='valid filename')

    parser.add_argument('--setting', default='evaluation', type=str, help='evaluation or inference')

    args = parser.parse_args()

    data = Data(args)




if __name__ == "__main__":
    main()
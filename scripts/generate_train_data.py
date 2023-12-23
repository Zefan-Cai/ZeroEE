import os
import copy
import json
import random
import argparse
from tqdm import tqdm


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

            # Add data for the parent
            if parent_event in self.data[parent_event]["data"].keys():

                event_name = self.data[parent_event]["data"][parent_event]["name"]
                samples = self.data[parent_event]["data"][parent_event]["samples"]

                diverse_definitions = copy.deepcopy(self.data[parent_event]["data"][parent_event]["rewrite_definitions"])
                event_definition = self.data[parent_event]["data"][parent_event]["definition"]
                diverse_definitions.insert(0, event_definition)
                diverse_definitions = diverse_definitions[:self.args.num_definitions]

                for sample in samples:

                    sentence = sample["sentence"]
                    definition = random.choice(diverse_definitions)
                    trigger = sample["trigger"]
                    # selected_trigger = random.choice(triggers)

                    if self.args.template_version == "v1":
                        prompt = f"{sentence} \n The event is: {event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                    elif self.args.template_version == "v2":
                        prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event_name}. \n DEFINITION: {definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                    else:
                        raise Exception("THe template version should be v1 or v2")

                    if type(prompt) == list: prompt = prompt[0]
                    positive_train_data.append({
                        "prompt": prompt[0],
                        "completion": f"Event trigger is {trigger}."
                        })

                    # Negative Sampling based on negative definitions
                    train_parent_list_without_parent = copy.deepcopy(self.train_parent_list)
                    train_parent_list_without_parent.remove(parent_event)
                    for index in range(self.args.num_negative_sample):
                        negative_parent_event = random.choice(train_parent_list_without_parent)
                        negative_event = random.choice(list(self.data[negative_parent_event]["data"].keys()))

                        negative_event_name = self.data[negative_parent_event]["data"][negative_event]["name"]

                        negative_diverse_definitions = copy.deepcopy(self.data[negative_parent_event]["data"][negative_event]["rewrite_definitions"])
                        negative_event_definition = self.data[negative_parent_event]["data"][negative_event]["definition"]
                        negative_diverse_definitions.append(negative_event_definition)
                        definition = random.choice(negative_diverse_definitions)

                        negative_text_sons = ", ".join(self.data[negative_parent_event]["sons"])

                        if self.args.template_version == "v1":
                            prompt = f"{sentence} \n The event is: {negative_event_name}. \n The event definition is: {definition} \n The parent event is {negative_parent_event}, son events include {negative_text_sons}. \n So what is the trigger?",
                        elif self.args.template_version == "v2":
                            prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {negative_event_name}. \n DEFINITION: {definition} \n PARENT: {negative_parent_event}, SON: {negative_text_sons}. \n So what is the trigger?",
                        else:
                            raise Exception("THe template version should be v1 or v2")

                        if type(prompt) == list: prompt = prompt[0]

                        negative_train_data.append({
                            "prompt": prompt[0],
                            "completion": f"Event trigger is <trigger>."
                            })

            # Add data for the son
            for event in sons:

                # Negative Sample
                negative_sons = copy.deepcopy(sons)
                if event in negative_sons:
                    negative_sons.remove(event)

                event_name = self.data[parent_event]["data"][event]["name"]
                # triggers = self.data[parent_event]["data"][event]["triggers"]
                samples = self.data[parent_event]["data"][event]["samples"]

                diverse_definitions = copy.deepcopy(self.data[parent_event]["data"][event]["rewrite_definitions"])
                event_definition = self.data[parent_event]["data"][event]["definition"]
                diverse_definitions.insert(0, event_definition)
                diverse_definitions = diverse_definitions[:self.args.num_definitions]

                # for definition in diverse_definitions:
                for sample in samples:

                    sentence = sample["sentence"]
                    trigger = sample["trigger"]
                    # selected_trigger = random.choice(triggers)

                    definition = random.choice(diverse_definitions)

                    if self.args.template_version == "v1":
                        prompt = f"{sentence} \n The event is: {event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                    elif self.args.template_version == "v2":
                        prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event_name}. \n DEFINITION: {definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                    else:
                        raise Exception("THe template version should be v1 or v2")

                    if type(prompt) == list: prompt = prompt[0]

                    positive_train_data.append({
                        "prompt": prompt[0],
                        "completion": f"Event trigger is {trigger}."
                        })

                    # Negative Sampling for in-ontology sons
                    if len(negative_sons) == 0: continue
                    for index in range(self.args.num_negative_inOntology):
                        negative_event = random.choice(negative_sons)

                        negative_event_name = self.data[parent_event]["data"][negative_event]["name"]

                        negative_diverse_definitions = copy.deepcopy(self.data[parent_event]["data"][negative_event]["rewrite_definitions"])
                        negative_event_definition = self.data[parent_event]["data"][negative_event]["definition"]
                        negative_diverse_definitions.append(negative_event_definition)
                        definition = random.choice(negative_diverse_definitions)

                        if self.args.template_version == "v1":
                            prompt = f"{sentence} \n The event is: {negative_event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                        elif self.args.template_version == "v2":
                            prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {negative_event_name}. \n DEFINITION: {definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                        else:
                            raise Exception("THe template version should be v1 or v2")

                        if type(prompt) == list: prompt = prompt[0]

                        negative_train_data.append({
                            "prompt": prompt[0],
                            "completion": f"Event trigger is <trigger>."
                            })


                    train_parent_list_without_parent = copy.deepcopy(self.train_parent_list)
                    train_parent_list_without_parent.remove(parent_event)
                    for index in range(self.args.num_negative_sample - self.args.num_negative_inOntology):
                        negative_parent_event = random.choice(train_parent_list_without_parent)
                        negative_event = random.choice(list(self.data[negative_parent_event]["data"].keys()))

                        negative_event_name = self.data[negative_parent_event]["data"][negative_event]["name"]

                        negative_diverse_definitions = copy.deepcopy(self.data[negative_parent_event]["data"][negative_event]["rewrite_definitions"])
                        negative_event_definition = self.data[negative_parent_event]["data"][negative_event]["definition"]
                        negative_diverse_definitions.append(negative_event_definition)
                        definition = random.choice(negative_diverse_definitions)

                        negative_text_sons = ", ".join(self.data[negative_parent_event]["sons"])

                        if self.args.template_version == "v1":
                            prompt = f"{sentence} \n The event is: {negative_event_name}. \n The event definition is: {definition} \n The parent event is {negative_parent_event}, son events include {negative_text_sons}. \n So what is the trigger?",
                        elif self.args.template_version == "v2":
                            prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {negative_event_name}. \n DEFINITION: {definition} \n PARENT: {negative_parent_event}, SON: {negative_text_sons}. \n So what is the trigger?",
                        else:
                            raise Exception("THe template version should be v1 or v2")

                        if type(prompt) == list: prompt = prompt[0]

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

            # Add data for the parent
            if parent_event in self.data[parent_event]["data"].keys():

                event_name = self.data[parent_event]["data"][parent_event]["name"]
                samples = self.data[parent_event]["data"][parent_event]["samples"]

                diverse_definitions = copy.deepcopy(self.data[parent_event]["data"][parent_event]["rewrite_definitions"])
                event_definition = self.data[parent_event]["data"][parent_event]["definition"]
                diverse_definitions.insert(0, event_definition)
                diverse_definitions = diverse_definitions[:self.args.num_definitions]

                event_id = self.valid_event_list.index(parent_event)

                for sample in samples:

                    sentence = sample["sentence"]
                    definition = random.choice(diverse_definitions)
                    trigger = sample["trigger"]
                    # selected_trigger = random.choice(triggers)

                    if self.args.template_version == "v1":
                        prompt = f"{sentence} \n The event is: {event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                    elif self.args.template_version == "v2":
                        prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event_name}. \n DEFINITION: {definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
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

                    # Negative Sampling based on negative definitions
                    valid_parent_list_without_parent = copy.deepcopy(self.valid_parent_list)
                    valid_parent_list_without_parent.remove(parent_event)
                    for index in range(self.args.num_negative_sample):
                        negative_parent_event = random.choice(valid_parent_list_without_parent)
                        negative_event = random.choice(list(self.data[negative_parent_event]["data"].keys()))

                        event_id = self.valid_event_list.index(negative_event)
                        negative_event_name = self.data[negative_parent_event]["data"][negative_event]["name"]

                        negative_diverse_definitions = copy.deepcopy(self.data[negative_parent_event]["data"][negative_event]["rewrite_definitions"])
                        negative_event_definition = self.data[negative_parent_event]["data"][negative_event]["definition"]
                        negative_diverse_definitions.append(negative_event_definition)
                        definition = random.choice(negative_diverse_definitions)

                        negative_text_sons = ", ".join(self.data[negative_parent_event]["sons"])

                        if self.args.template_version == "v1":
                            prompt = f"{sentence} \n The event is: {negative_event_name}. \n The event definition is: {definition} \n The parent event is {negative_parent_event}, son events include {negative_text_sons}. \n So what is the trigger?",
                        elif self.args.template_version == "v2":
                            prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {negative_event_name}. \n DEFINITION: {definition} \n PARENT: {negative_parent_event}, SON: {negative_text_sons}. \n So what is the trigger?",
                        else:
                            raise Exception("THe template version should be v1 or v2")

                        if type(prompt) == list: prompt = prompt[0]

                        negative_valid_data.append({
                            "data_id": data_index,
                            "event_type": event_id,
                            "prompt": prompt[0],
                            "completion": f"Event trigger is ",
                            "trigger": "<trigger>"
                            })

                    data_index += 1



            for event in sons:
                # Negative Sample
                negative_sons = copy.deepcopy(sons)
                if event in negative_sons:
                    negative_sons.remove(event)

                event_name = self.data[parent_event]["data"][event]["name"]
                event_definition = self.data[parent_event]["data"][event]["definition"]
                samples = self.data[parent_event]["data"][event]["samples"]

                diverse_definitions = copy.deepcopy(self.data[parent_event]["data"][event]["rewrite_definitions"])
                event_definition = self.data[parent_event]["data"][event]["definition"]
                diverse_definitions.insert(0, event_definition)
                diverse_definitions = diverse_definitions[:self.args.num_definitions]

                # for evaluation
                event_id = self.valid_event_list.index(event)
                for sample in samples:
                    sentence = sample["sentence"]
                    trigger = sample["trigger"]
                    definition = random.choice(diverse_definitions)

                    if self.args.template_version == "v1":
                        prompt = f"{sentence} \n The event is: {event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                    elif self.args.template_version == "v2":
                        prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event_name}. \n DEFINITION: {definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
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

                    # Negative Sampling for in-ontology sons
                    for index in range(len(negative_sons)):
                        negative_event = negative_sons[index]

                        event_id = self.valid_event_list.index(negative_event)
                        negative_event_name = self.data[parent_event]["data"][negative_event]["name"]

                        negative_diverse_definitions = copy.deepcopy(self.data[parent_event]["data"][negative_event]["rewrite_definitions"])
                        negative_event_definition = self.data[parent_event]["data"][negative_event]["definition"]
                        negative_diverse_definitions.append(negative_event_definition)
                        definition = random.choice(negative_diverse_definitions)

                        if self.args.template_version == "v1":
                            prompt = f"{sentence} \n The event is: {negative_event_name}. \n The event definition is: {definition} \n The parent event is {parent_event}, son events include {text_sons}. \n So what is the trigger?",
                        elif self.args.template_version == "v2":
                            prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {negative_event_name}. \n DEFINITION: {definition} \n PARENT: {parent_event}, SON: {text_sons}. \n So what is the trigger?",
                        else:
                            raise Exception("THe template version should be v1 or v2")

                        if type(prompt) == list: prompt = prompt[0]
                        negative_valid_data.append({
                            "data_id": data_index,
                            "event_type": event_id,
                            "prompt": prompt[0],
                            "completion": f"Event trigger is ",
                            "trigger": f"<trigger>"
                            })

                    valid_parent_list_without_parent = copy.deepcopy(self.valid_parent_list)
                    valid_parent_list_without_parent.remove(parent_event)
                    for index in range(self.args.num_negative_sample - self.args.num_negative_inOntology):
                        negative_parent_event = random.choice(valid_parent_list_without_parent)
                        negative_event = random.choice(list(self.data[negative_parent_event]["data"].keys()))

                        event_id = self.valid_event_list.index(negative_event)
                        negative_event_name = self.data[negative_parent_event]["data"][negative_event]["name"]

                        negative_diverse_definitions = copy.deepcopy(self.data[negative_parent_event]["data"][negative_event]["rewrite_definitions"])
                        negative_event_definition = self.data[negative_parent_event]["data"][negative_event]["definition"]
                        negative_diverse_definitions.append(negative_event_definition)
                        definition = random.choice(negative_diverse_definitions)

                        negative_text_sons = ", ".join(self.data[negative_parent_event]["sons"])

                        if self.args.template_version == "v1":
                            prompt = f"{sentence} \n The event is: {negative_event_name}. \n The event definition is: {definition} \n The parent event is {negative_parent_event}, son events include {negative_text_sons}. \n So what is the trigger?",
                        elif self.args.template_version == "v2":
                            prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {negative_event_name}. \n DEFINITION: {definition} \n PARENT: {negative_parent_event}, SON: {negative_text_sons}. \n So what is the trigger?",
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
    
    parser.add_argument('--num_negative_sample', default=1, type=int, help='')
    parser.add_argument('--num_negative_inOntology', default=1, type=int, help='')
    parser.add_argument('--num_definitions', default=1, type=int, help='')
    parser.add_argument('--random_seed', default=1, type=int, help='')

    parser.add_argument('--train_parent_start', default=50, type=int, help='')
    parser.add_argument('--train_parent_end', default=100, type=int, help='')
    parser.add_argument('--valid_parent_start', default=0, type=int, help='')
    parser.add_argument('--valid_parent_end', default=50, type=int, help='')

    args = parser.parse_args()


    data = Data(args)
    random.seed(args.random_seed)




if __name__ == "__main__":
    main()
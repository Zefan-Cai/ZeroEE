'''
Author: JustBluce 972281745@qq.com
Date: 2024-02-01 14:48:47
LastEditors: JustBluce 972281745@qq.com
LastEditTime: 2024-02-01 15:00:50
FilePath: /ZeroEE/ZeroEE/scripts_v2/generate_GENEVA_train_data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



import os
import random
import json
import argparse
from tqdm import tqdm


# GENEVA




class Data():

    def __init__(self, args):

        self.args = args

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        with open(os.path.join(self.args.base_dir, 'ZeroEE', 'data', self.args.event_definition_dir), 'r') as fp:
            self.event_type2definition = json.load(fp)

        # 获得event type的列表，列表的key是event sample的数量，列表的value是event type的列表
        with open(os.path.join(self.args.base_dir, 'ZeroEE', 'data', 'times2events.json'), 'r') as fp:
            times2events = json.load(fp)

        self.event_list = times2events[self.args.least_sampleNum_perEvent] # 拿到完整96个evemt的list
        # 和ACE overlap的event的list
        self.overlap_events = ['Telling', 'Arrest', 'Social_event', 'Come_together', 'Supply', 'Getting', 'Commerce_sell', 'Giving', 'Commerce_buy', 'Earnings_and_losses', 'Receiving', 'Exchange', 'Commerce_pay', 'Death', 'Bodily_harm', 'Protest', 'Communication', 'Traveling', 'Attack']

        self.number_of_events = len(self.event_list)
        self.available_events = self.event_list

        # 获得96个event的event type到event dēfinition的映射
        self.avalibale_event_type2definition = {}
        for key, item in self.event_type2definition.items():
            if key in self.available_events:
                self.avalibale_event_type2definition[key] = item

        self.GENEVA_training_data = []
        with open(os.path.join(self.args.base_dir, 'ZeroEE', 'GENEVA-main', 'data', 'train.json'), 'r') as fp:
            for line in fp.readlines():
                self.GENEVA_training_data.append(json.loads(line))

        self.get_train_data()

    def get_train_data(self):

        for num_sample in [1, 5, 10, 20, 40, 2000]:


            for n_negative in [self.args.num_negative_sample]:

                if n_negative > self.number_of_events: continue

                positive_train_data = []
                negative_train_data = []
                train_data = []

                event2times = {}

                for index in range(len(self.GENEVA_training_data)):

                    sentence = self.GENEVA_training_data[index]["sentence"]

                    self.GENEVA_training_data[index]["sentence"] = self.GENEVA_training_data[index]["sentence"].replace("\'\'", "")

                    event_type2trigger = {}
                    for event_index in range(len(self.GENEVA_training_data[index]["event_mentions"])):

                        event_type = self.GENEVA_training_data[index]["event_mentions"][event_index]["event_type"]

                        if event_type in self.available_events and event_type not in self.overlap_events:
                            trigger = self.GENEVA_training_data[index]["event_mentions"][event_index]["trigger"]["text"]

                            if event_type not in event2times.keys():
                                event2times[event_type] = 0

                            if event2times[event_type] <= num_sample:
                                if event_type not in event_type2trigger.keys():
                                    event_type2trigger[event_type] = []
                                event_type2trigger[event_type].append(trigger)
                                event2times[event_type] += 1


                    for event_type in event_type2trigger.keys():
                        event_definitions = self.avalibale_event_type2definition[event_type]
                        event_definitions = event_definitions[:self.args.num_definitions]

                        for event_definition in event_definitions:

                            if self.args.template_version == "v1":
                                prompt = f"{sentence} \n The event is: {event_type}. \n "
                                if self.args.add_definition:
                                    prompt += f"The event definition is: {event_definition} \n "
                                prompt += "So what is the trigger?"
                            elif self.args.template_version == "v2":
                                prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event_type}. \n "
                                if self.args.add_definition:
                                    prompt += f"DEFINITION: {event_definition} \n "
                                prompt += "So what is the trigger?"
                            else:
                                raise Exception("THe template version should be v1 or v2")

                            positive_train_data.append({
                                "prompt": prompt,
                                "completion": "Event trigger is {}".format(" and ".join(event_type2trigger[event_type]))
                                })


                    if event_type2trigger != {}:

                        available_evet_types = list(set(self.available_events) - set(event_type2trigger.keys()))
                        if len(available_evet_types) > n_negative:
                            selected_event_type = random.sample(available_evet_types, n_negative)
                        else:
                            selected_event_type = random.sample(available_evet_types, len(available_evet_types))

                        for event_type in selected_event_type:
                            event_definitions = self.avalibale_event_type2definition[event_type]
                            event_definitions = event_definitions[:self.args.num_definitions]

                            for event_definition in event_definitions:

                                if self.args.template_version == "v1":
                                    prompt = f"{sentence} \n The event is: {event_type}. \n "
                                    if self.args.add_definition:
                                        prompt += f"The event definition is: {event_definition} \n "
                                    prompt += "So what is the trigger?"
                                elif self.args.template_version == "v2":
                                    prompt = f"SENTENCE: {sentence} \n EVENT TYPE: {event_type}. \n "
                                    if self.args.add_definition:
                                        prompt += f"DEFINITION: {event_definition} \n "
                                    prompt += "So what is the trigger?"
                                else:
                                    raise Exception("THe template version should be v1 or v2")

                                negative_train_data.append({
                                    "prompt": prompt,
                                    "completion": "Event trigger is <trigger>"
                                    })

                train_data = positive_train_data + negative_train_data

                output_file_name = f'GENEVA_train_negatives{str(n_negative)}_samples{str(num_sample)}_events{str(self.number_of_events)}'
                if self.args.add_definition:
                    output_file_name+= f"_{str(self.args.num_definitions)}definition"
                else:
                    output_file_name+= "_0definition"
                output_file_name += ".jsonl"
                with open(os.path.join(self.args.base_dir, 'data', self.args.output_dir, output_file_name), 'w') as fp:
                    for line in train_data:
                        json.dump(line, fp)
                        fp.write('\n')












def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--service', default='plus', type=str, help='plus or pku')
    parser.add_argument('--template_version', default='v2', type=str, help='v1 or v2')

    parser.add_argument('--base_dir', default="/local1/zefan/", type=str, help=' plus: /local1/zefan/   pku: /home/caizf/projects/ZeroEE/ ')
    parser.add_argument('--event_definition_dir', default='GENEVA_event_definition.json', type=str, help='dir to event definition data. Geneva: GENEVA_event_definition.json; ACE: Geneva_ToAce_event_definition.json')
    parser.add_argument('--output_dir', default='geneva_train_v2_data', type=str, help='dir to output data')
    parser.add_argument('--least_sampleNum_perEvent', default='0', type=str, help='dir to generated data')

    parser.add_argument('--add_definition', default=False, type=bool, help='')
    parser.add_argument('--num_definitions', default=1, type=int, help='')

    parser.add_argument('--num_negative_sample', default=1, type=int, help='')
    parser.add_argument('--random_seed', default=1, type=int, help='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    data = Data(args)


if __name__ == "__main__":
    main()
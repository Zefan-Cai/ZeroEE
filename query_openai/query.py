# Load Original Event Definition

import json
from tqdm import tqdm

ev_def_file = "Event_Definition/GENEVA_event_definition.json"
ev_def_data = json.load(open(ev_def_file, "r"))
ev_def_file_para = "output/Geneva_para_event_def.json"

output_dict = {}
for ev_type, ev_def in ev_def_data.items():
    output_dict[ev_type] = {
        "Query": f"Please rephrase the following passage in 5 other formats in separate lines. \nPassage: \"{ev_def}\"",
        "Response": ""
    }
with open(ev_def_file_para, "w") as F:
    json.dump(output_dict, F, indent = 4)

import openai
openai.api_key = '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'

# Load Data
ChatGPTQueryResponseDict = json.load(open(ev_def_file_para, "r"))

try_num = 1000 # run all
processed = 0
for task, QueryResponse in tqdm(ChatGPTQueryResponseDict.items()):
    if processed == try_num:
        print(f"Stop with {processed} iterations at task: {task}")
        break
    if QueryResponse['Response'] == "": # Not processed yet
        reply = None
        while reply is None:
            try:
                print(f"Query task: {task}")
                messages = [ {"role": "user", "content": QueryResponse['Query']} ]
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages
                )
                reply = chat.choices[0].message.content
                ChatGPTQueryResponseDict[task]['Response'] = reply
                processed += 1
                with open(ev_def_file_para, "w") as F:
                    json.dump(ChatGPTQueryResponseDict, F, indent=2)
            except Exception as e:
                print(f"Error occured at task: {task}.")
                print(e)
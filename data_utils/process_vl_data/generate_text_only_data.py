import os
import json

import copy

save_dict = []

for i in range(0,45):
    for j in range(0,45):
        for id in range(1):
            x={}
            if id==0:
                question = f"Solve the equation and find the correct answer of {str(i)} plus {str(j)}."
                answer = f"{str(i)} plus {str(j)} equals {str(i+j)}."
            else:
                question = f"{str(i)} + {str(j)}=?"
                answer = str(i+j)

            template = [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
            x["id"] = 'text_'+str(i)+'_'+str(j)+'_'+str(id)
            x["conversations"] = template
            save_dict.append(x)

save_dir = '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_text_only_plus_0412.json'
json.dump(save_dict, open(save_dir, 'w'), indent=4)
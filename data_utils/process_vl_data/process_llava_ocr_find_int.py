import copy
import json
def find_integers(s):
    integers = []
    current_number = ''
    for char in s:
        if char.isdigit():
            current_number += char
        elif current_number:
            integers.append(int(current_number))
            current_number = ''
    if current_number:
        integers.append(int(current_number))
    return integers

json_path = '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/0315_llava_ocr_has_number.json'
with open(json_path) as f:
    data = json.load(f)
new_data = []
for d in data:
    new_d = copy.deepcopy(d)
    new_conversations = []
    for i,conversation in enumerate(d['conversations']):
        if i==0:
            new_conversations.append({'from':'human','value':'<image>\nSolve the question: point out the number.'})
        elif i%2==0:
            new_conversations.append({'from':'human','value':'Solve the question: point out the number.'})
        else:
            integers = find_integers(conversation['value'])
            integers = [str(i) for i in integers]
            integers_str = ','.join(integers)
            if len(integers)>1:
                ans=f'There are {len(integers)} number displays below: '+integers_str
            else:
                ans='There is a number displays below: '+integers[0]
            new_conversations.append({'from':'gpt','value':ans})
    new_d['conversations'] = copy.deepcopy(new_conversations)
    new_data.append(new_d)

print(len(new_data))
save_path = '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_llava_ocr_has_number_0316.json'
json.dump(new_data, open(save_path, "w"), indent=4)
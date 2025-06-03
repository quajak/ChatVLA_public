import copy
import json
import os
json_path = "/home/jz08/zhouzy/data/vl_data/llava_finetune_data/llava_ocr.json"
with open(json_path) as f:
    data = json.load(f)
print(len(data))
keywords = ['0','1','2','3','4','5','6','7','8','9']
save_keywords = "_".join(keywords)
# save_path = json_path.replace("_ocr.json", "_{}.json".format(save_keywords))
save_path = json_path.replace("_ocr.json", "_ocr_has_math.json")

# text_json = []
# text_index = []
# for i,item in enumerate(data):
#     try:
#         d=item['image']
#     except:
#         text_index.append(i)
#         text_json.append(item)
# data = [d for i, d in enumerate(data) if i not in text_index]
# print(len(data))
# json.dump(text_json, open(json_path.replace('llava_v1_5_mix665k.json','llava_text_only.json'), 'w'), ensure_ascii=False, indent=4)

# has_key_in_path = [d for d in data if any(k in d['image'] for k in keywords)]
has_key_in_conversation = [d for d in data if any(k in d['conversations'][id]['value'] for id in range(len(d['conversations'])) for k in keywords)]
print(len(has_key_in_conversation))
for d in has_key_in_conversation:
    conv_list = []
    id_list_ans = {id for id in range(len(d['conversations'])) if any(k in d['conversations'][id]['value'] for k in keywords)}
    for i,ans in enumerate(id_list_ans):
        if ans==1:
            tmp = copy.deepcopy(d['conversations'][ans - 1])
            tmp['value'] = tmp['value'].replace('<image>\n','<image>\nSolve the question: ')
            conv_list.append(tmp)
        elif i!=0:
            tmp = copy.deepcopy(d['conversations'][ans - 1])
            tmp['value'] = 'Solve the question: ' + tmp['value']
            conv_list.append(tmp)
        else:
            tmp = copy.deepcopy(d['conversations'][ans-1])
            tmp['value'] = '<image>\nSolve the question: '+tmp['value']
            conv_list.append(tmp)

        conv_list.append(d['conversations'][ans])

    d['conversations'] = copy.deepcopy(conv_list)
json.dump(has_key_in_conversation, open(save_path, "w"), indent=4)

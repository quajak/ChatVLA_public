import os.path
import h5py
import os
from tqdm import tqdm
import json
import random
import copy
def generate_answer_or_reasoning(task_config):
    with open(task_config['template_path'], 'r') as f:
        task_template = json.load(f)['math']
    random_list = [random.randint(0, len(task_template[i]) - 1) for i in
                   range(len(task_template))]
    sub_reasoning = [task_template[i][t] for i, t in enumerate(random_list)]
    sub_reasoning = ' '.join(sub_reasoning)
    for i in task_config['placeholders']:
        sub_reasoning = sub_reasoning.replace(str(i), str(task_config['placeholders'][i]))
    group_id = int(task_config['id'])
    sub_reasoning = sub_reasoning.replace('<pos>',
                                          str(task_config['roll_placeholders']['pos'][group_id // 2]))
    tmp_list = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]]
    sub_reasoning = sub_reasoning.replace('<ans1>',
                                          str(task_config['roll_placeholders']['answer'][tmp_list[group_id][0]]))
    sub_reasoning = sub_reasoning.replace('<ans2>',
                                          str(task_config['roll_placeholders']['answer'][
                                                  tmp_list[group_id][1]]))
    sub_reasoning = sub_reasoning.replace('<ans3>',
                                          str(task_config['roll_placeholders']['answer'][
                                                  tmp_list[group_id][2]]))
    return sub_reasoning

raw_json_dir='/home/jz08/zhouzy/data/vl_data/vl_ocr_three_below.json'
with open(raw_json_dir, 'r') as f:
    raw_data = json.load(f)
save_dict=[]

for d in tqdm(raw_data):
    ############# for reason aug
    # config = {
    #     'id': d['id'][-1],
    #     'placeholders': {
    #         '<A>': d['all_numbers'][0],
    #         '<B>': d['all_numbers'][1],
    #         '<C>': d['all_numbers'][2],
    #         '<op>': '+'
    #     },
    #     'roll_placeholders': {
    #         'answer': [d['all_numbers'][2], d['all_numbers'][3], d['all_numbers'][4]],
    #         'pos': ['left', 'middle', 'right'],
    #     },
    #     'template_path': '/home/jz08/zhouzy/code/moevla/data_utils/generate_reasoning/using_reasoning_same.json',
    # }
    # question = 'Solve the equation and point out the correct answer.'
    # answer = generate_answer_or_reasoning(config)


    # ########## for single ocr
    # ans = d['all_numbers']
    # if d['id'].endswith('_0'):
    #     question = 'Solve the question: point out the number on the gray foam pad.'
    #     answer = f'On the gray foam is {ans}.'
    # elif d['id'].endswith('_1'):
    #     question = 'Solve the question: point out the number below the gray foam pad.'
    #     answer = f'There is a number displays below: {ans} (left).'
    # else:
    #     question = 'Solve the question: point out the number below the gray foam pad.'
    #     answer = f'There is a number displays below: {ans} (right).'


    ############ for multiple ocr
    num_list = d['all_numbers'].split('_')
    question = 'Solve the question: point out the number.'
    answer = f'There are three numbers display below: {num_list[0]} (left), {num_list[1]} (middle), and {num_list[2]} (right).'
    template = [
        {
            "from": "human",
            "value": f"<image>\n{question}"
        },
        {
            "from": "gpt",
            "value": answer
        }
    ]
    x = copy.deepcopy(d)
    x["conversations"] = template
    save_dict.append(x)


save_dir = '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_ocr_three_below_0317.json'
json.dump(save_dict, open(save_dir, 'w'), indent=4)
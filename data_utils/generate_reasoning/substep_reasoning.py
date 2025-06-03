import os.path
import h5py
import os
from tqdm import tqdm
import json
import random




def generate_reasoning_replace_placeholder(TASK_CONFIG,data_dir):
    save_reasoning_data = {}
    for task_id,task in enumerate(TASK_CONFIG):
        print("<"*20+"processing task {}".format(task)+">"*20)
        task_config = TASK_CONFIG[task]
        with open(task_config['template_path'], 'r') as f:
            task_template = json.load(f)['math']

        file_path = os.path.join(data_dir, task, f"{task}_succ_t0001_s-0-0")
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        random.seed(42+task_id)
        save_reasoning_data[task] = {}

        for file_id, file_name in tqdm(enumerate(file_list[:])):
            random_list = [random.randint(0, len(task_template[i]) - 1) for i in
                           range(len(task_template))]
            sub_reasoning = [task_template[i][t] for i,t in enumerate(random_list)]
            sub_reasoning = ' '.join(sub_reasoning)
            for i in task_config['placeholders']:
                sub_reasoning = sub_reasoning.replace(str(i), str(task_config['placeholders'][i]))
            group_id = file_id//8
            sub_reasoning = sub_reasoning.replace('<pos>',
                                                  str(task_config['roll_placeholders']['pos'][group_id // 2]))
            tmp_list = [[0,1,2],[0,2,1],[1,0,2],[2,0,1],[1,2,0],[2,1,0]]
            sub_reasoning = sub_reasoning.replace('<ans1>',
                                                  str(task_config['roll_placeholders']['answer'][tmp_list[group_id][0]]))
            sub_reasoning = sub_reasoning.replace('<ans2>',
                                                  str(task_config['roll_placeholders']['answer'][
                                                          tmp_list[group_id][1]]))
            sub_reasoning = sub_reasoning.replace('<ans3>',
                                                  str(task_config['roll_placeholders']['answer'][
                                                          tmp_list[group_id][2]]))
            sub_reasoning=[str(sub_reasoning)]
            file_path = str(file_path)
            try:
                deal_subtask_label(path=file_path, sub_reasonings=sub_reasoning, file_name=file_name,raw_lang=task_config['raw_lang'])
                save_reasoning_data[task][file_name] = sub_reasoning
            except Exception as e:
                print(e)
                print(task, file_name)
                exit(1)

    save_reasoning_dir = '/home/jovyan/tzb/zhouzy/code/moevla/data_utils/saved_data_reasoning_json/math_mani_0313.json'
    json.dump(save_reasoning_data, open(save_reasoning_dir, 'w'), indent=4)

def deal_subtask_label(path:str='/Volumes/PSSD-8/ljm/5_28_put_rubbish_can_succ_t0001_s-0-0/episode_12.hdf5',sub_reasonings=None, file_name=None,save_path=None, raw_lang=None)->None:
    data = h5py.File(os.path.join(path,file_name),'a')
    # convert zero value of substep to max number of substeps (only for the last of the list)
    substep_index = data['substeps'][:]
    maxnum = max(substep_index)
    idx = len(substep_index) - 1
    while(substep_index[idx] != maxnum):
        substep_index[idx] = maxnum
        idx -= 1



    if "language_raw" not in data.keys():
        data.create_dataset("language_raw", data=[raw_lang])
    else:
        # data.attrs['language_raw'] = data['language_raw'][()]
        data['language_raw'][()] = [raw_lang]
    if sub_reasonings is not None:
        targets = [sub_reasonings[i] for i in substep_index]
        if "substep_reasonings" not in data.keys():
            data.create_dataset("substep_reasonings", data=targets)
        else:
            data.attrs['substep_reasonings'] = data['substep_reasonings'][()]
            data['substep_reasonings'][()] = targets

    data.close()


if __name__=="__main__":
    # DATA_DIR='/media/jz08/ADDS-4/mobile_franka_data/0224_plus_4_task'
    # template_path = '/home/jz08/zhouzy/code/tinyvla_vl_data_zzy/data_utils/generate_reasoning/using_reasoning_type.json'

    DATA_DIR='/home/jovyan/tzb/h5py_data/mobile_franka_data'
    template_path = '/home/jovyan/tzb/zhouzy/code/moevla/data_utils/generate_reasoning/using_reasoning_same_v2.json'
    task_config = {
        '0224_1_plus_4_480_640': {
            'raw_lang':'Solve the equation and move to the correct answer.',
            'placeholders':{
                '<A>':1,
                '<B>':4,
                '<C>':5,
                '<op>':'+'
            },
            'roll_placeholders':{
                'answer':[5,0,6],
                'pos':['left','middle','right'],
            },
            'template_path':template_path,
        },
        '0224_1_plus_7_480_640': {
            'raw_lang': 'Solve the equation and move to the correct answer.',
            'placeholders': {
                '<A>': 1,
                '<B>': 7,
                '<C>': 8,
                '<op>': '+'
            },
            'roll_placeholders': {
                'answer': [8, 9, 0],
                'pos': ['left', 'middle', 'right'],
            },
            'template_path': template_path,
        },
        '0224_3_plus_6_480_640': {
            'raw_lang': 'Solve the equation and move to the correct answer.',
            'placeholders': {
                '<A>': 3,
                '<B>': 6,
                '<C>': 9,
                '<op>': '+'
            },
            'roll_placeholders': {
                'answer': [9, 2, 4],
                'pos': ['left', 'middle', 'right'],
            },
            'template_path': template_path,
        },
        '0224_2_plus_5_480_640': {
            'raw_lang': 'Solve the equation and move to the correct answer.',
            'placeholders': {
                '<A>': 2,
                '<B>': 5,
                '<C>': 7,
                '<op>': '+'
            },
            'roll_placeholders': {
                'answer': [7, 8, 3],
                'pos': ['left', 'middle', 'right'],
            },
            'template_path': template_path,
        }
    }
    generate_reasoning_replace_placeholder(task_config,DATA_DIR)
    exit(0)

    # ######### load from json #################
    DATA_DIR = '/home/jovyan/tzb/h5py_data/mobile_franka_data'
    existing_data_json = '/home/jovyan/tzb/zhouzy/code/moevla/data_utils/saved_data_reasoning_json/math_mani_0313.json'
    with open(existing_data_json, 'r') as f:
        reasoning_data = json.load(f)
    task_list = list(reasoning_data.keys())
    for trsr in task_list:
        file_path = os.path.join(DATA_DIR, trsr, f"{trsr}_succ_t0001_s-0-0")
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))

        for file_name in tqdm(file_list[:]):
            sub_reasoning = reasoning_data[trsr][file_name]
            try:
                deal_subtask_label(path=file_path, sub_reasonings=sub_reasoning, file_name=file_name,raw_lang='Solve the equation and move to the correct answer.')

            except Exception as e:
                print(e)
                print(trsr, file_name)
    exit(0)

    ########### adding reasoning randomly ################
    DATA_DIR = '/home/jovyan/tzb/wjj/data/mobile_franka_data/'
    reasoning_path = '/home/jovyan/tzb/zhumj/code/tinyvla_vl_data_zzy/data_utils/generate_reasoning/used_reasoning.json'
    with open(reasoning_path, 'r') as f:
        reasoning_data = json.load(f)
    task_list = list(raw_lang.keys())
    # task_list = ["0114_yellow_pink_building_block_to_basket_480_640", "0114_open_drawer_put_spider_man_480_640","0114_stack_building_block_480_640","0115_bussing_table_two_animals_two_cubes_480_640"]
    task_list.sort()
    save_reasoning_data = {}
    for trsr in task_list:
        file_path = os.path.join(DATA_DIR, trsr, f"{trsr}_succ_t0001_s-0-0")
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        reasoning_data_for_task = reasoning_data[trsr]
        random.seed(42)
        save_reasoning_data[trsr] = {}
        for file_name in tqdm(file_list[:]):
            if isinstance(reasoning_data_for_task[0], list):
                random_list = [random.randint(0, len(reasoning_data_for_task[i])-1) for i in range(len(reasoning_data_for_task))]

                sub_reasoning = [reasoning_data_for_task[i][random_list[i]] for i in range(len(reasoning_data_for_task))]
            else:
                sub_reasoning = [reasoning_data_for_task[random.randint(0, len(reasoning_data_for_task)-1)]]

            try:
                deal_subtask_label(path=file_path, sub_reasonings=sub_reasoning, file_name=file_name,raw_lang=raw_lang[trsr])
                save_reasoning_data[trsr][file_name] = sub_reasoning
            except Exception as e:
                print(e)
                print(trsr, file_name)
    save_reasoning_dir = reasoning_path.replace('used_reasoning.json','mobile_franka_reasoning_w_vl_data_selected.json')
    json.dump(save_reasoning_data, open(save_reasoning_dir, 'w'), indent=4)


    # for trsr, sub_r in task_require_substep_reasonings.items():
    #     file_path = os.path.join(DATA_DIR, trsr, f"{trsr}_succ_t0001_s-0-0")
    #     # file_path="/home/jz08/zhouzy/data/mobile_franka_data/1228_toy_pig"
    #     file_list = os.listdir(file_path)
    #     file_list = [f for f in file_list if f.endswith('.hdf5')]
    #     file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    #     for file_name in tqdm(file_list[:]):
    #         try:
    #             deal_subtask_label(path=file_path, sub_reasonings=sub_r,file_name=file_name,raw_lang=raw_lang[trsr])
    #         except Exception as e:
    #             print(e)
    #             print(trsr, file_name)

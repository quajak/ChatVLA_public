import copy
import os.path
import time
import h5py
import numpy as np
from tqdm import tqdm
import sys
# sys.path.append('/gpfs/private/tzb/wjj/projects/dvla_mh_qwen2_vla')
import os
from tqdm import tqdm


def down_sample(path, step):
    data=h5py.File(path,'a')

    stack = [(data)]

    while stack:
        cur_data = stack.pop()
        for name, item in cur_data.items():
            if isinstance(item, h5py.Group):
                stack.append((item))
            else:
                shape=item[:].shape[0]
                print(shape)
                # if item[:].shape[0] > 40:
                #     # sample_data = item[0::step]
                #     sample_data = item[:int(shape/2)]
                #     del cur_data[name]
                #     cur_data.create_dataset(name, data=sample_data)


    data.close()

if __name__=="__main__":

    DATA_DIR = '/home/jovyan/tzb/h5py_data/mobile_franka_data/'
    ignore=[]

    # ignore = ['1029_place_cup_on_the_shelf', '1030_hide_spiderman', '1030_magic_cube', '1031_sweep_trash', '1031_unpack_bag_put_ball', '1030_put_light_bulb']

    # OUTPUT=os.path.join(DATA_DIR)
    # files = os.listdir(DATA_DIR)
    files = ['0224_1_plus_4_480_640_modified','0224_1_plus_7_480_640_modified','0224_2_plus_5_480_640_modified','0224_3_plus_6_480_640_modified']
    for file in tqdm(files):
        file_path = os.path.join(DATA_DIR, file, file+'_succ_t0001_s-0-0')
        if not os.path.isdir(os.path.join(DATA_DIR, file)):
            continue
        if file in ignore:
            continue
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        i = 0
        for file_name in tqdm(file_list[:]):
            down_sample(os.path.join(file_path, file_name), step=3)
            i += 1


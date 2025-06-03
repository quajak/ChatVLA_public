import h5py
import os
from constants import TASK_CONFIGS
import cv2
import numpy as np
import json

def save_actions_to_file(h5_file_path, output_file_path):
    """
    Save all actions from an HDF5 file to a JSON file.
    Args:
        h5_file_path: Path to the HDF5 file
        output_file_path: Path where to save the actions (will be saved as .json)
    """
    # Make sure the output file has .json extension
    output_file_path = os.path.splitext(output_file_path)[0] + '.json'
    
    with h5py.File(h5_file_path, 'r') as data:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        actions_data = {
            "metadata": {
                "source_file": h5_file_path,
                "total_actions": 0,
                "timestamp": None  # You can add timestamp if needed
            },
            "actions": []
        }
        
        # Get actions data
        if 'action' in data:
            actions = data['action'][:]
            # actions = actions.astype(np.float16)
            actions_data["metadata"]["total_actions"] = len(actions)
            actions_data["actions"]=actions.tolist()
        else:
            actions_data["metadata"]["total_actions"] = 0
            
        # Save to JSON file
        with open(output_file_path, 'w') as f:
            json.dump(actions_data, f, indent=4)

save_folder='/home/jovyan/tzb/zhouzy/data/zzy_check_data_0419/'
# os.makedirs(save_folder, exist_ok=True)
task_name='0413_mix_math'
# task_name='mobile_franka_reasoning_w_vl_data_selected'
task_name='local_debug_data_zzy'
save_folder= '/home/jz08/zhouzy/data/zzy_check_data'
os.makedirs(save_folder, exist_ok=True)
folder_list = TASK_CONFIGS[task_name]['dataset_dir']
# folder_list = ['/home/jovyan/tzb/h5py_data/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0/']
folder_list.sort()
for folder_path in folder_list:
    # if task_name=='mobile_franka_reasoning_w_vl_data_selected':
    #     print(folder_path.split('/')[-2])
    if 'local_debug_data' in task_name:
        try:
            last_name = folder_path.split('/')[-1]
            file_list = os.listdir(os.path.join(folder_path, last_name + '_succ_t0001_s-0-0'))
            file_list = [f for f in file_list if f.endswith('.hdf5')]
            file_list.sort()
            h5_path = os.path.join(folder_path, last_name + '_succ_t0001_s-0-0', file_list[0])
            data = h5py.File(h5_path, 'r')
            # print(data['language_raw'][:])
            print(data['substep_reasonings'][0])
            #
            # Save actions to JSON file
            # output_path = os.path.join(folder_path, last_name + '_succ_t0001_s-0-0', 'actions')
            # save_actions_to_file(h5_path, output_path)
        except Exception as e:
            print(e)
            pass
    else:

        try:
            last_name = folder_path.split('/')[-1]
            print("#############         Processing " +last_name + "           ############")
            file_list = os.listdir(folder_path)
            file_list = [f for f in file_list if f.endswith('.hdf5')]
            file_list.sort()
            h5_path = os.path.join(folder_path, file_list[0])
            data = h5py.File(h5_path, 'r')
            print(data['language_raw'][0])
            print(data['substep_reasonings'][0])
            # Save actions to JSON file
            output_path = os.path.join(save_folder, 'left_actions')
            save_actions_to_file(h5_path, output_path)

            # all_data = data['observations']['images']['left'][:]
            #
            # # image_data = all_data[0]
            # # cv2.imwrite(os.path.join(save_folder,last_name+'.png'), image_data)
            # os.makedirs(os.path.join(save_folder, last_name), exist_ok=True)
            # for idx in range(0,all_data.shape[0]):
            #     image_data = all_data[idx][:,:,::-1]
            #     cv2.imwrite(os.path.join(save_folder, last_name, str(idx) + '.png'), image_data)
            # print("#############  " +last_name + " SUCCESS           ############")
        except:
            print("#############  " + last_name + " FAILED           ############")
            print(folder_path.split('/')[-1])




# name='0110_put_blue_laundry_detergent_into_box_480_640'
# folder_path = os.path.join('/home/jovyan/tzb/wjj/data/mobile_franka_data',name,name+'_succ_t0001_s-0-0')
# file_list = os.listdir(folder_path)
# file_list = [f for f in file_list if f.endswith('.hdf5')]
# print("#### TOTAL NUMBER ###     ",len(file_list))
# file_list.sort()
# for filename in file_list[:10]:
#     print(filename)
#     # filename = '/home/jovyan/tzb/wjj/data/mobile_franka_data/0106_bread_to_empty_plate_480_640/0106_bread_to_empty_plate_480_640_succ_t0001_s-0-0/episode_9.hdf5'
#     data = h5py.File(os.path.join(folder_path, filename), 'r')
#     print(data['substeps'][:])
#     # print(data['substep_reasonings'][:])
#     print(data['language_raw'][:])
#     # exit(0)
#     # for key in data.keys():
#     #     print(key)
#     #     try:
#     #         print(data[key][:].shape)
#     #     except:
#     #         print("skip")
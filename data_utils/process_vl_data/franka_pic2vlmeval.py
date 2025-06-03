import json

import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.image
import csv

json_keys = ['ocr_val','two_digits_plus_15_20_for_val','added_in_10_answer_over_10_for_val','single_ocr_for_val', 'two2one_for_val']
# json_keys = ['two2one_2_for_val']
save_pic_dir = f'/home/jz08/zhouzy/data/vl_data/processed_real_world'
os.makedirs(save_pic_dir, exist_ok=True)


data = [['index','question','A','B','C','D','answer','category','abcLabel','image_path']]
q_list = [
    # 'Solve the equation and point out the correct answer.',
    # 'Solve the question: point out the number below the gray foam pad.',
    'What is the color of the foam?',
    'Is there an apple?',
    'How many numbers are there?'
]
q_text_list = []
# for i in range(20,31):
#     for j in range(20,31):
#         q_text_list.append(f'Solve the equation and find the correct answer of {i} plus {j}.')
cnt=0
use_left_view = False
use_wrist_view = True
save_dataset_dir = '/home/jz08/LMUData/images/MMRO_mini_0408_wrist'
os.makedirs(save_dataset_dir, exist_ok=True)


for k in json_keys:
    save_names = []
    ############ save image ###############
    raw_json_dir=f'/home/jz08/zhouzy/data/vl_data/vl_{k}.json'
    with open(raw_json_dir, 'r') as f:
        raw_data = json.load(f)
    for d in raw_data:
        image_data = []
        image_data_left = []
        image_data_wrist = []
        for p in d['image']:
            img = np.array(Image.open(os.path.join('/home/jz08/zhouzy/data/vl_data', p)))
            # img = np.array(Image.open(os.path.join('/home/jz08/zhouzy/data/vl_data', p)).convert("RGB"))
            img = cv2.resize(img, (320, 240))
            image_data.append(img)
            if 'left' in p:
                image_data_left.append(img)
            if 'wrist' in p:
                image_data_wrist.append(img)
        image_data = np.concatenate(image_data,axis=0)
        image_data_left = np.array(image_data_left[0])
        image_data_wrist = np.array(image_data_wrist[0])

        if use_left_view:
            left_name = k + d['all_numbers'] + '_' + d['id'][-1] + '_left.jpg'
            save_names.append(left_name)
            # cv2.imwrite(os.path.join(save_pic_dir, left_name), image_data_left)
            # cv2.imwrite(os.path.join(save_dataset_dir, left_name), image_data_left)
        elif use_wrist_view:
            wrist_name = k + d['all_numbers'] + '_' + d['id'][-1] + '_wrist.jpg'
            save_names.append(wrist_name)
            # cv2.imwrite(os.path.join(save_pic_dir, wrist_name), image_data_wrist)
            # cv2.imwrite(os.path.join(save_dataset_dir, wrist_name), image_data_wrist)
        else:
            full_name = k + d['all_numbers'] + '_' + d['id'][-1] + '_3_views.jpg'
            save_names.append(full_name)
            # cv2.imwrite(os.path.join(save_pic_dir, full_name), image_data)
            # cv2.imwrite(os.path.join(save_dataset_dir, full_name), image_data)


    
    ############ write to csv ################
    for n in save_names:
        for q in q_list:
            data.append([str(cnt), q, '', '', '', '', '', k, '',n])
            cnt+=1
for q in q_text_list:
    data.append([str(cnt), q, '', '', '', '', '', 'text_math', '', ''])
    cnt += 1
filename = '/home/jz08/LMUData/MMRO_mini_0420.tsv'

# Writing to tsv file
with open(filename, 'a', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(data)

print(f"TSV file '{filename}' created successfully.")
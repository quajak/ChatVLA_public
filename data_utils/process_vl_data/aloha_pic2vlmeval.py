import json

import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.image
import csv



data = [['index','question','A','B','C','D','answer','category','abcLabel','image_path']]
q_list = [
    'Describe the image.'
    # 'Answer the question and pick the card with the correct answer.',
    # 'Answer the question and pick the card with the correct answer.',
    # 'Answer the question and pick the card with the correct answer.',
    # 'Answer the question and pick the card with the correct answer.',
    # 'Answer the question and pick the card with the correct answer.'
]

cnt=0

keys=['plus','minus','times','div']
for n in range(1,13):
    img_name = 'img_' + str(n) + '.jpg'
    for q in q_list:
        data.append([str(cnt), q, '', '', '', '', '', 'describe_'+keys[cnt//12], '',img_name])
        cnt+=1
for n in range(25,61):
    img_name = 'img_' + str(n) + '.jpg'
    for q in q_list:
        data.append([str(cnt), q, '', '', '', '', '', 'describe_'+keys[cnt//12], '',img_name])
        cnt+=1

filename = '/home/jz08/LMUData/MMRO_mini_describe.tsv'

# Writing to tsv file
with open(filename, 'a', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(data)

print(f"TSV file '{filename}' created successfully.")
import os
import cv2
from attr.filters import exclude

# folder_path = '/home/jz08/zhouzy/data/vl_data'
# folder_path = '/home/jz08/zhouzy/data/vl_data'
folder_path = '/home/jovyan/tzb/zhouzy/data/LMUData/images'
save_folder_path = folder_path
# save_folder_path = '/home/jz08/zhouzy/data/vl_data/verification_code_resize'
os.makedirs(save_folder_path, exist_ok=True)
folder_list = os.listdir(folder_path)
folder_list.sort()
exclude_list = [
]
# folder_list = ['verification_code']
folder_list = ['MMRO_mini']
for f in folder_list:
    if f not in exclude_list and os.path.isdir(folder_path + '/' + f):
        print(f)
        img_list = os.listdir(folder_path + '/' + f)

        for img_name in img_list:
            if img_name.endswith('.jpg') and len(img_name) < 9:
                img = cv2.imread(os.path.join(folder_path + '/' + f, img_name))
                img = img[:, :, ::-1]
                # img = cv2.resize(img, (320, 240))
                cv2.imwrite(os.path.join(save_folder_path + '/' + f, img_name), img)
                # cv2.imwrite(os.path.join(save_folder_path, img_name), img)
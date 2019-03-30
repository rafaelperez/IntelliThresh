import os

import numpy as np
import cv2

from intellithresh import IntelliThresh

data_root = '/home/mscv1/Desktop/FRL/ChengleiSocial_full/ChengleiSocial/undistorted'
frame_id  = '020453'

bg_id = '000000'

output_dir = '/home/mscv1/Desktop/FRL/ChengleiSocial_full/ChengleiSocial/masks'

frame_list = os.listdir(data_root)

assert bg_id in frame_list and frame_id in frame_list

bg_frame_dir = os.path.join(data_root, bg_id)
target_frame_dir = os.path.join(data_root, frame_id)

view_list = os.listdir(target_frame_dir)

ignore_views = {
    '400183',
    '400196',
    '400204',
    '400219',
    '410141'
}

masker = IntelliThresh()

for view_id in view_list:
    try:
        if view_id in ignore_views:
            continue
        
        if view_id[1] == '1':
            continue

        img_path = os.path.join(target_frame_dir, view_id)
        bg_path = os.path.join(bg_frame_dir, view_id)

        img = cv2.imread(img_path, 1)
        bg = cv2.imread(bg_path, 1)

        mask = masker.get_mask(img, bg)
        
        output_path = os.path.join(output_dir, frame_id)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, view_id)
        cv2.imwrite(output_path, mask)
        print('Wrote to {}'.format(output_path))
    except Exception:
        print('View {} failed!'.format(view_id))
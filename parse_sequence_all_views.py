import cv2
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

from utils import crop_out, filter_largest_component
from masker.deeplab_pytorch.api import DeepLabV2JointBKSMasker

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script for generating mask on cluster.')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--frame_begin', type=str, required=True)
    parser.add_argument('--frame_end', type=str, required=True)
    parser.add_argument('--pid', type=int, default=-1)
    args = parser.parse_args()

    print('process {} Setting up masker...'.format(args.pid))
    masker = DeepLabV2JointBKSMasker(crf=False)

    # test_set_tag = 'chenglei_social_full'
    data_root = args.data_root
    bg_dir = '000000'
    frame_format = '%06d'
    frame_begin = '020450'
    frame_end = '021000'
    view_id = '400170'

    exp_tag = 'deeplabv2jointbks_seq'

    output_dir = 'output'
    output_folder = str(datetime.datetime.now()).replace(' ', '_')
    output_folder_path = os.path.join(output_dir, output_folder)
    output_folder_path += ('_' + exp_tag)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for frame_id_int in range(int(frame_begin), int(frame_end)+1, 1):
        frame_id = frame_format % frame_id_int

        img_path = os.path.join(data_root, frame_id, view_id + '.png')
        bk_path = os.path.join(data_root, bg_dir, view_id + '.png')
        print('Doing {}...'.format(img_path))

        try:
            img = cv2.imread(img_path, 1)
            bk = cv2.imread(bk_path, 1)
        except Exception:
            print('Error when parsing {}'.format(img_path))
            continue

        # init_fg_mask = fg_marker.infer_fg(img)
    
        # learned_mask = masker.get_mask(img, bk, init_fg_mask)
        learned_mask = masker.get_mask(img, bk)

        # trimap = get_trimap_from_raw_mask_basic(learned_mask)

        # alpha = do_matting(img, trimap)

        # learned_mask = alpha
        learned_mask = filter_largest_component(learned_mask)

        demo_img = crop_out(img, learned_mask)

        output_mask_name = '{}_{}_mask.png'.format(frame_id, view_id)
        output_demo_name = '{}_{}_demo.png'.format(frame_id, view_id)
        # output_tri_name = '{}_{}_tri.png'.format(frame_id, view_id)

        output_mask_path = os.path.join(output_folder_path, output_mask_name)
        output_demo_path = os.path.join(output_folder_path, output_demo_name)
        # output_tri_path = os.path.join(output_folder_path, output_tri_name)
        
        cv2.imwrite(output_mask_path, learned_mask)
        cv2.imwrite(output_demo_path, demo_img)
        # cv2.imwrite(output_tri_path, trimap)
        print('Wrote to {}'.format(output_demo_path))




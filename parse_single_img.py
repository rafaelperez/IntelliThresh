import cv2
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

from utils import crop_out, filter_largest_component
from masker.deeplab_pytorch.api import DeepLabV2JointBKSV2Masker
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get mask for single image. Need background.')
    parser.add_argument('--img_path', type=str, required=True, help='path to image')
    parser.add_argument('--out_path', type=str, required=True, help='path to output image')
    parser.add_argument('--bg_dir', type=str, required=True, help='path to background frame folder')
    args = parser.parse_args()

    print('Setting up masker...')
    masker = DeepLabV2JointBKSV2Masker(crf=False)

    # test_set_tag = 'chenglei_social_full'
    # data_root = '/home/mscv1/Desktop/FRL/ChengleiSocial_full/ChengleiSocial/undistorted'
    # bg_dir = '000000'
    # frame_format = '%06d'
    # frame_begin = '020450'
    # frame_end = '021000'
    # view_id = '400170'

    # exp_tag = 'deeplabv2jointbks_seq'

    # output_dir = 'output'
    # output_folder = str(datetime.datetime.now()).replace(' ', '_')
    # output_folder_path = os.path.join(output_dir, output_folder)
    # output_folder_path += ('_' + exp_tag)

    # if not os.path.exists(output_folder_path):
        # os.makedirs(output_folder_path)

    img_path = args.img_path
    
    img_path_spt = img_path.split('/')
    img_id = img_path_spt[-1]
    bk_path = os.path.join(args.bg_dir, img_id)
    print('Doing {}...'.format(img_path))

    img = cv2.imread(img_path, 1)
    bk = cv2.imread(bk_path, 1)

    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bk = cv2.cvtColor(bk, cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bk = cv2.cvtColor(bk, cv2.COLOR_GRAY2BGR)
    """
    

    # bounding_box = (1128, 2588, 1992, 3600) # [l, [t, r), b)
    # cropped image will still follow the rescale rule specified by CONFIG
    # where max edge will be scaled to CONFIG.TEST.SIZE

    # l, t, r, b = bounding_box
    # img = img[t:b, l:r, :]
    # bk = bk[t:b, l:r, :]

    # init_fg_mask = fg_marker.infer_fg(img)

    # learned_mask = masker.get_mask(img, bk, init_fg_mask)
    learned_mask = masker.get_mask(img, bk)

    # trimap = get_trimap_from_raw_mask_basic(learned_mask)

    # alpha = do_matting(img, trimap)

    # learned_mask = alpha
    learned_mask = filter_largest_component(learned_mask)

    # demo_img = crop_out(img, learned_mask)

    # output_mask_name = '{}_{}_mask.png'.format(frame_id, view_id)
    # output_demo_name = '{}_{}_demo.png'.format(frame_id, view_id)
    # output_tri_name = '{}_{}_tri.png'.format(frame_id, view_id)

    # output_mask_path = os.path.join(output_folder_path, output_mask_name)
    # output_demo_path = os.path.join(output_folder_path, output_demo_name)
    # output_tri_path = os.path.join(output_folder_path, output_tri_name)
    
    cv2.imwrite(args.out_path, learned_mask)
    # cv2.imwrite(output_demo_path, demo_img)
    # cv2.imwrite(output_tri_path, trimap)
    # print('Wrote to {}'.format(output_demo_path))
    print('Wrote to {}'.format(args.out_path))




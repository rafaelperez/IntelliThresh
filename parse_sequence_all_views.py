import cv2
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

from utils import crop_out, filter_largest_component
from masker.deeplab_pytorch.api import DeepLabV2JointBKSMasker
from multiprocessing import Pool

import torch

import argparse

import torch.multiprocessing as multiprocessing

def do_parsing(param_dict):
    frame_begin_num, frame_end_num = param_dict['frame_begin_num'], param_dict['frame_end_num']
    
    print(frame_begin_num)
    print(frame_end_num)

    if frame_begin_num >= frame_end_num:
        return

    nid, pid = param_dict['nid'], param_dict['pid']
    data_root = param_dict['data_root']
    output_root = param_dict['output_root']
    bg_dir = param_dict['bg_dir']

    print('node {} process {} Setting up masker...'.format(nid, pid))
    masker = DeepLabV2JointBKSMasker(crf=False, device_id=pid)

    frame_format = '%06d'
    bg_path = os.path.join(data_root, bg_dir)

    for frame_id_int in range(frame_begin_num, frame_end_num, 1):
        frame_id = frame_format % frame_id_int
        frame_path = os.path.join(data_root, frame_id)
        
        frame_output_path = os.path.join(output_root, frame_id)
        if not os.path.exists(frame_output_path):
            os.makedirs(frame_output_path)
        
        all_views = os.listdir(frame_path)
        all_views.sort()

        for view_id in all_views:
            img_path = os.path.join(frame_path, view_id)
            bk_path = os.path.join(bg_path, view_id)
            print('Node {} Process {} Doing {}...'.format(nid, pid, img_path))

            img = cv2.imread(img_path, 1)
            bk = cv2.imread(bk_path, 1)

            learned_mask = masker.get_mask(img, bk)

            learned_mask = filter_largest_component(learned_mask)

            # demo_img = crop_out(img, learned_mask)

            output_mask_name = '{}.png'.format(view_id)
            # output_demo_name = '{}_{}_demo.png'.format(frame_id, view_id)

            output_mask_path = os.path.join(frame_output_path, output_mask_name)
            # output_demo_path = os.path.join(output_folder_path, output_demo_name)
            
            cv2.imwrite(output_mask_path, learned_mask)
            # cv2.imwrite(output_demo_path, demo_img)
            print('Node {} Process {} Wrote to {}'.format(nid, pid, output_mask_path))



if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Script for generating mask on cluster.')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--frame_begin', type=str, required=True)
    parser.add_argument('--frame_end', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--node_id', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=1)
    parser.add_argument('--bg_dir', type=str, default='000000')
    args = parser.parse_args()

    # print('process {} Setting up masker...'.format(args.pid))
    # masker = DeepLabV2JointBKSMasker(crf=False)

    # test_set_tag = 'chenglei_social_full'
    data_root = args.data_root
    bg_dir = args.bg_dir
    frame_begin = args.frame_begin
    frame_end = args.frame_end
    # view_id = '400170'

    exp_tag = 'deeplabv2jointbks_seq'

    output_root_path = args.output_dir
    
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    frame_begin_num = int(frame_begin) # inclusive
    frame_end_num = int(frame_end)+1 # exclusive

    if args.node_num > 1:
        total_frame_num = frame_end_num - frame_begin_num
        frame_num_per_node = (total_frame_num + args.node_num - 1) // args.node_num
        cur_node_frame_begin_num = frame_begin_num + args.node_id * frame_num_per_node
        cur_node_frame_end_num = min(cur_node_frame_begin_num + frame_num_per_node, frame_end_num)
    else:
        cur_node_frame_begin_num = frame_begin_num
        cur_node_frame_end_num = frame_end_num

    if cur_node_frame_begin_num < cur_node_frame_end_num:
        process_num = torch.cuda.device_count()
        print(process_num)

        process_pool = Pool(process_num)
        cur_node_total_frame_num = cur_node_frame_end_num - cur_node_frame_begin_num
        frame_per_process = (cur_node_total_frame_num + process_num - 1 ) // process_num
        param_list = list()
        for i in range(process_num):
            param_dict = dict()
            param_dict['nid'] = args.node_id
            param_dict['pid'] = i
            param_dict['data_root'] = data_root
            param_dict['bg_dir'] = bg_dir
            param_dict['frame_begin_num'] = cur_node_frame_begin_num + i * frame_per_process
            param_dict['frame_end_num'] = min(cur_node_frame_end_num, param_dict['frame_begin_num'] + frame_per_process)
            param_dict['output_root'] = output_root_path
            param_list.append(param_dict)
            
        process_pool.map(do_parsing, param_list)
    else:
        print('Node {} Finished'.format(args.node_id))



    




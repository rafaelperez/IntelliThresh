import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from gen_img import show_img

from multiprocessing import Pool

'''
dome_frames_root = '/home/mscv1/Desktop/FRL/ChengleiSocial/undistorted'
frame_folders = os.listdir(dome_frames_root)
bg_folder = '000000'
frame_folders = [folder for folder in frame_folders if folder != bg_folder]
frame_num = len(frame_folders)
'''

bg_folder_path = '/home/mscv1/Desktop/FRL/dome_bg_imgs/real_bg_imgs/bkimggreen'

view_imgs = os.listdir(bg_folder_path)
view_num = len(view_imgs)


def resize_img(img, new_size):
    new_h, new_w = new_size
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return new_img
    

def gen_patches(param):
    pid, num = param
    print('process {} started'.format(pid))

    patch_count = 0
    crop_size = 1284
    save_size = 642
    crop_per_img = 5

    root = '/home/mscv1/Desktop/FRL/synthetic_bgs/bg_patches'
    
    if not osp.exists(root):
        try:
            os.makedirs(root)
        except Exception:
            pass

    while patch_count < num:
        for k in range(crop_per_img):
            view_id = random.choice(view_imgs)
            bg_img_path = osp.join(bg_folder_path, view_id)
            bg_img = cv2.imread(bg_img_path, cv2.IMREAD_COLOR)
            
            # Cropping
            h, w = bg_img.shape[:2]
            start_h = random.randint(0, h - crop_size)
            start_w = random.randint(0, w - crop_size)
            end_h = start_h + crop_size
            end_w = start_w + crop_size
            bg_patch = bg_img[start_h:end_h, start_w:end_w, :]

            bg_patch = cv2.resize(bg_patch, (save_size, save_size))

            save_path = osp.join(root, 'patch_{}_{}.png'.format(pid, patch_count))

            cv2.imwrite(save_path, bg_patch)
            print('process {} saved to {}'.format(pid, save_path))

            patch_count += 1
        


if __name__ == '__main__':
    syn_bg_num = 32
    num_process = 16
    work_per_process = syn_bg_num // num_process
    p = Pool(num_process)
    p.map(gen_patches, [(pid, work_per_process) for pid in range(num_process)])
    
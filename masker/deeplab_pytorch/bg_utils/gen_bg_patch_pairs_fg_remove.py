"""
This script generate 'real' and 'synthetic' backround pairs for 
deep background removal network. 'synthetic' bg patches are generated
by simply removing forground object in a forground image taken from the same
view point. 'real' bg patches are croped from pure bg images. 
"""

import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from gen_img import show_img

from multiprocessing import Pool

dome_frames_root = '/home/mscv1/Desktop/FRL/ChengleiSocial/undistorted'
frame_folders = os.listdir(dome_frames_root)
bg_folder = '000000'
frame_folders = [folder for folder in frame_folders if folder != bg_folder]
frame_num = len(frame_folders)

bg_folder_path = osp.join(dome_frames_root, bg_folder)

view_imgs = os.listdir(bg_folder_path)
view_num = len(view_imgs)


def resize_img(img, new_size):
    new_h, new_w = new_size
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return new_img

def simple_human_remove(fg, bg):
    fgd = resize_img(fg, (800, 400)).astype(np.float32)
    bgd = resize_img(bg, (800, 400)).astype(np.float32)
    diff = np.sum((fgd-bgd)**2, axis=2)**0.5
    diff = resize_img(diff, fg.shape[:2])
    mask = diff > 30

    # plt.imshow(mask)
    # plt.show()

    output = fg.copy()
    for i in range(fg.shape[2]):
        output[:, :, i][mask] = bg[:, :, i][mask]
    return output

def get_syn_bg():
    chose_frame_id = np.random.randint(low=0, high=frame_num)
    chose_frame_folder = frame_folders[chose_frame_id]
    chose_view_id = np.random.randint(low=0, high=view_num)
    chose_view_img = view_imgs[chose_view_id]
    is_gray = chose_view_img[:2] == '41'
    
    fg_img_path = osp.join(
        dome_frames_root, 
        chose_frame_folder,
        chose_view_img
    )

    bg_img_path = osp.join(
        dome_frames_root,
        bg_folder,
        chose_view_img
    )

    fg_img = cv2.imread(fg_img_path, cv2.IMREAD_COLOR)
    bg_img = cv2.imread(bg_img_path, cv2.IMREAD_COLOR)

    syn_bg = simple_human_remove(fg_img, bg_img)

    # show_img(syn_bg)
    return syn_bg, bg_img, is_gray


def gen_patches(param):


    pid, num = param
    print('process {} started'.format(pid))

    patch_count = 0
    crop_size = 1284
    save_size = 642
    crop_per_img = 5

    root = '/home/mscv1/Desktop/FRL/synthetic_bgs'
    syn_path = osp.join(root, 'syn_bgs')
    real_path = osp.join(root, 'real_bgs')
    
    if not osp.exists(syn_path):
        os.makedirs(syn_path)
    
    if not osp.exists(real_path):
        os.makedirs(real_path)

    while patch_count < num:
        for k in range(crop_per_img):
            syn_bg, real_bg, is_gray = get_syn_bg()
            # Cropping
            h, w = syn_bg.shape[:2]
            start_h = random.randint(0, h - crop_size)
            start_w = random.randint(0, w - crop_size)
            end_h = start_h + crop_size
            end_w = start_w + crop_size
            syn_patch = syn_bg[start_h:end_h, start_w:end_w, :]
            real_patch = real_bg[start_h:end_h, start_w:end_w, :]

            syn_patch = cv2.resize(syn_patch, (save_size, save_size))
            real_patch = cv2.resize(real_patch, (save_size, save_size))

            if is_gray:
                save_id = '{}_{}_gray.jpg'.format(pid, patch_count)
            else:
                save_id = '{}_{}_rgb.jpg'.format(pid, patch_count)

            save_path_syn = osp.join(syn_path, save_id)
            save_path_real = osp.join(real_path, save_id)

            cv2.imwrite(save_path_syn, syn_patch)
            print('process {} saved to {}'.format(pid, save_path_syn))
            cv2.imwrite(save_path_real, real_patch)
            print('process {} saved to {}'.format(pid, save_path_real))

            patch_count += 1
        


if __name__ == '__main__':
    syn_bg_num = 50000
    num_process = 16
    work_per_process = syn_bg_num // num_process
    p = Pool(num_process)
    p.map(gen_patches, [(pid, work_per_process) for pid in range(num_process)])
    
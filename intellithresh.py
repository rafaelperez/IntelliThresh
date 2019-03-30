import cv2
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

from utils import crop_out, filter_largest_component

from human_parser.openpose_pytorch.api import OpenPoseForeGroundMarker
from trimap_generator import get_trimap_from_raw_mask_basic
# from matting.closed_form_matting.api import do_matting
from matting.grabcut.api import do_matting

from masker.deep_diff.api import VGG16DeepDiffMasker
# from masker.flownet2_diff.api import FlowNet2DiffMasker
from masker.baseline.api import FRLBaselineMasker
from masker.deeplab_pytorch.api import DeepLabV2Masker
from masker.deeplab_pytorch.api import DeepLabV2JointBKSMasker

class IntelliThresh(object):
    def __init__(self):
        print('Setting up human parsers...')
        self.fg_marker = OpenPoseForeGroundMarker()
        print('Setting up image descriptor...')
        self.masker = VGG16DeepDiffMasker()

    def get_mask(self, img, bg):
        init_fg_mask = self.fg_marker.infer_fg(img)
    
        learned_mask = self.masker.get_mask(img, bg, init_fg_mask)

        trimap = get_trimap_from_raw_mask_basic(learned_mask)

        alpha = do_matting(img, trimap)

        return alpha


if __name__ == '__main__':

    print('Setting up human parsers...')
    # fg_marker = OpenPoseForeGroundMarker()
    print('Setting up image descriptor...')
    # masker = VGG16DeepDiffMasker()
    # masker = FlowNet2DiffMasker()
    # masker = FRLBaselineMasker()
    # masker = DeepLabV2Masker()
    masker = DeepLabV2JointBKSMasker(crf=True)

    import test_list_chenglei_social_full as test_set
    
    test_set_tag = test_set.test_set_tag
    data_root = test_set.data_root
    test_img_list = test_set.test_img_list
    bg_dir = test_set.bg_dir

    attrs_of_interest = ['color', 'hole', 'gray']
    exp_tag = 'deeplabv2jointbks_crf_color_2048'

    output_dir = 'output'
    output_folder = str(datetime.datetime.now()).replace(' ', '_')
    output_folder_path = os.path.join(output_dir, output_folder)
    output_folder_path += ('_' + exp_tag)
    for attr in attrs_of_interest:
        output_folder_path += ('_' + attr)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    attrs_of_interest = set(attrs_of_interest)

    for item in test_img_list:
        frame_id, view_id, attrs = item
        do_this = False
        for attr in attrs_of_interest:
            if attr in attrs:
                do_this = True
                break
        if not do_this:
            continue

        img_path = os.path.join(data_root, frame_id, view_id + '.png')
        bk_path = os.path.join(data_root, bg_dir, view_id + '.png')
        print('Doing {}...'.format(img_path))


        img = cv2.imread(img_path, 1)
        bk = cv2.imread(bk_path, 1)

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




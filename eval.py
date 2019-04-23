import cv2
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

from utils import crop_out, filter_largest_component

from trimap_generator import get_trimap_from_fine_mask
from matting.closed_form_matting.api import do_matting
# from matting.grabcut.api import do_matting

from masker.deep_diff.api import VGG16DeepDiffMasker
# from masker.flownet2_diff.api import FlowNet2DiffMasker
from masker.baseline.api import FRLBaselineMasker
from masker.deeplab_pytorch.api import DeepLabV2Masker
from masker.deeplab_pytorch.api import DeepLabV2JointBKSMasker, DeepLabV2JointBKSV2Masker, DeepDiffE2EMasker

from cv2.ximgproc import guidedFilter


if __name__ == '__main__':

    print('Setting up human parsers...')
    # fg_marker = OpenPoseForeGroundMarker()
    print('Setting up image descriptor...')
    # masker = VGG16DeepDiffMasker()
    # masker = FlowNet2DiffMasker()
    # masker = FRLBaselineMasker()
    # masker = DeepLabV2Masker()
    # masker = DeepLabV2JointBKSMasker(crf=True)
    masker = DeepLabV2JointBKSV2Masker(crf=False)
    # masker = DeepDiffE2EMasker()

    
    from val_set import get_samples

    samples = get_samples()

    exp_tag = 'val_raw_alpha'
    output_dir = 'output'
    output_folder = str(datetime.datetime.now()).replace(' ', '_')
    output_folder_path = os.path.join(output_dir, output_folder)
    output_folder_path += ('_' + exp_tag)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for item in samples:
        img_path, bk_path, sample_id, gt_path = item

        # print('Doing {}...'.format(img_path))

        img = cv2.imread(img_path, 1)
        bk = cv2.imread(bk_path, 1)

        learned_mask = masker.get_mask(img, bk)

        # trimap = get_trimap_from_fine_mask(learned_mask)
        # alpha = do_matting(img, trimap)
        # learned_mask = alpha
        if '41' in sample_id:
            guide_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
            print(guide_img.shape)
        else:
            guide_img = img.copy()

        learned_mask_dst = learned_mask.copy().astype(np.float32)
        guidedFilter(guide=guide_img.astype(np.float32), src=learned_mask.astype(np.float32), dst=learned_mask_dst, radius=30, eps=0.01)
        print(np.mean(learned_mask_dst))
        learned_mask = learned_mask_dst.clip(0, 255).astype(np.uint8)


        demo_img = crop_out(img, learned_mask)

        output_mask_name = '{}_mask.png'.format(sample_id)
        output_demo_name = '{}_demo.png'.format(sample_id)

        output_mask_path = os.path.join(output_folder_path, output_mask_name)
        output_demo_path = os.path.join(output_folder_path, output_demo_name)
        
        cv2.imwrite(output_mask_path, learned_mask)
        cv2.imwrite(output_demo_path, demo_img)
        # print('Wrote to {}'.format(output_demo_path))
        



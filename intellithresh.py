import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import torch

from vgg import vgg16_bn_descriptor
from utils import downsize_rgb, img_transform, crop_out


class DeepBkSubtractor(object):
    def __init__(self, max_size=1000, rgb_downscale_factor=0.2):
        self.max_size = max_size
        self.rgb_downscale_factor = rgb_downscale_factor
        self.device = torch.device('cuda:0')
        self.vgg16 = vgg16_bn_descriptor()
        self.vgg16.to(self.device)
        self.vgg16.eval()

    def get_rgb_diff(self, img, bg, factor):
        h, w = img.shape[:2]
        img_feature = downsize_rgb(img, factor)
        bg_feature = downsize_rgb(bg, factor)
        diff = np.sum((img_feature - bg_feature)**2, axis=2)**0.5
        diff = cv2.resize(diff, (w, h))
        return diff

    def get_diffs(self, img, bk):
        h, w = img.shape[0:2]

        diff_rgb = self.get_rgb_diff(img, bk, self.rgb_downscale_factor)

        img_trans = img_transform(img)
        bk_trans = img_transform(bk)

        batch = np.stack((img_trans, bk_trans), axis=0)
        batch = torch.tensor(batch, device=self.device)

        batch_features = self.vgg16(batch)
        diffs = []

        for batch_feature in batch_features:
            img_feature = batch_feature[0, :, :, :]
            bk_feature = batch_feature[1, :, :, :]
            
            diff = torch.sum((img_feature - bk_feature)**2, dim=0)
            
            diff = diff.detach().cpu().numpy()
            diff = cv2.resize(diff, (w, h))

            diffs.append(diff)

        diffs.append(diff_rgb)
        diffs = np.stack(diffs, axis=2)
        # whiten
        means = np.mean(diffs, axis=(0, 1), keepdims=True)
        stds = np.std(diffs, axis=(0, 1), keepdims=True)
        diffs = (diffs - means) / stds
        return diffs

    def get_mask(self, img, bk):
        diffs = self.get_diffs(img, bk)
        
        # solve threashold model from disparity feature
        # we need an initial positive mask.
        # TODO: call OpenPose here
        pos_mask = cv2.imread('pose_mask.png', 0)
        learned_mask = np.zeros(img.shape[0:2], dtype=np.uint8)

        for i in range(10):
            neg_mask = 255 - pos_mask

            pos_is, pos_js = np.where(pos_mask > 0)
            neg_is, neg_js = np.where(neg_mask > 0)

            pos_num = pos_is.shape[0]
            neg_num = neg_is.shape[0]

            neg_sample_ids = np.random.choice(a=neg_num, size=pos_num, replace=True)
            neg_is = neg_is[neg_sample_ids]
            neg_js = neg_js[neg_sample_ids]

            pos_feats = diffs[pos_is, pos_js, :]
            neg_feats = diffs[neg_is, neg_js, :]
            
            pos_feats = np.concatenate((pos_feats, np.ones((pos_num, 1), dtype=np.float32)), axis=1)
            neg_feats = np.concatenate((neg_feats, np.ones((pos_num, 1), dtype=np.float32)), axis=1)

            feats = np.concatenate((pos_feats, neg_feats), axis=0)

            labels = np.zeros((2*pos_num, 1), dtype=np.float32)
            labels[:pos_num, :] = 1

            weights = np.linalg.inv(feats.transpose() @ feats) @ (feats.transpose() @ labels)
            
            print(weights)

            h, w, c = diffs.shape
            score = diffs.reshape(h*w, c) @ weights[0:c] + weights[c]
            score = score.reshape(h, w)

            learned_mask_fgs = score > 0.5
            learned_mask.fill(0)
            learned_mask[learned_mask_fgs] = 255

            plt.imshow(learned_mask)
            plt.show()

            pos_mask = learned_mask

        return learned_mask
        
        
        
        


if __name__ == '__main__':
    data_root = '/home/mscv1/Desktop/FRL/ChengleiSocial/undistorted'
    bg_dir = '000000'
    # 020296
    #   400126
    #   400038

    # 020290
    #   400166
    #   

    # 020315
    #   400219

    img_dir = '020293'

    view_id = '400128'

    img = cv2.imread(os.path.join(data_root, img_dir, view_id + '.png'))
    bk = cv2.imread(os.path.join(data_root, bg_dir, view_id + '.png'))

    masker = DeepBkSubtractor()
    learned_mask = masker.get_mask(img, bk)

    demo_img = crop_out(img, learned_mask)
    plt.imshow(demo_img[:, :, ::-1])
    plt.show()
    




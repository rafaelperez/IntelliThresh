import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class VGGDescriptor(nn.Module):

    def __init__(self, features):
        super(VGGDescriptor, self).__init__()
        self.features = features
        self.keep_ids = {1, 4, 8, 12}

    def forward(self, x):
        feats = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.keep_ids:
                feats.append(x)
        return feats


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'D_descriptor': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'D_descriptor1': [64, 64]
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def vgg16_bn_descriptor():
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGDescriptor(make_layers(cfg['D_descriptor'], batch_norm=True))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']), strict=False)
    return model

def img_transform(img, max_edge=1000):
    h, w = img.shape[0:2]
    max_size = max(h, w)
    factor = max_edge / max_size
    nh, nw = int(h * factor), int(w * factor)
    img = cv2.resize(img, (nw, nh))

    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, -1))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, -1))
    img = (img - mean) / std
    img = np.transpose(img, [2, 0, 1])
    return img

def filter_largest_component(mask):
    output = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_16U)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    id_area_pair = [(stats[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, num_labels)]
    id_area_pair.sort(key=lambda x: -x[0])
    max_area_id = id_area_pair[0][1]
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    new_mask[labels == max_area_id] = 255

    new_mask_flood = new_mask.copy()
    cv2.floodFill(image=new_mask_flood, seedPoint=(0, 0), newVal=255, mask=None)
    new_mask = new_mask_flood - new_mask
    new_mask = 255 - new_mask

    return new_mask

def downsize_rgb(img, factor):
    h, w = img.shape[:2]
    nh, nw = int(h*factor), int(w*factor)
    img_rgb = cv2.resize(img, (nw, nh)).astype(np.float32)
    return img_rgb

def get_rgb_diff(img, bg, factor):
    h, w = img.shape[:2]
    img_feature = downsize_rgb(img, factor)
    bg_feature = downsize_rgb(bg, factor)
    diff = np.abs(img_feature - bg_feature)
    diff = cv2.resize(diff, (w, h))
    return diff

def get_rgb_diff_reduced(img, bg, factor):
    h, w = img.shape[:2]
    img_feature = downsize_rgb(img, factor)
    bg_feature = downsize_rgb(bg, factor)
    diff = np.sum((img_feature - bg_feature)**2, axis=2)**0.5
    diff = cv2.resize(diff, (w, h))
    return diff

class DeepBkSubstractor(object):
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.device = torch.device('cuda:0')
        self.vgg16 = vgg16_bn_descriptor()
        self.vgg16.to(self.device)
        self.vgg16.eval()

    def get_mask(self, img, bk):
        h, w = img.shape[0:2]
        img_trans = img_transform(img)
        bk_trans = img_transform(bk)

        batch = np.stack((img_trans, bk_trans), axis=0)
        batch = torch.tensor(batch, device=self.device)

        batch_features = self.vgg16(batch)
        diffs = []

        for batch_feature in batch_features:
            img_feature = batch_feature[0, :, :, :]
            bk_feature = batch_feature[1, :, :, :]
            
            diff = torch.sum((img_feature - bk_feature)**2, dim=0)**0.5
            
            diff = diff.detach().cpu().numpy()
            diff = cv2.resize(diff, (w, h))

            diffs.append(diff)

        diff0, diff1, diff2, diff3 = diffs

        """
        #
        bg_mask = diff1 < 0.6
        plt.imshow(bg_mask)
        plt.show()
        #
        """

        mask0 = np.zeros(diff0.shape, dtype=np.uint8)
        mask1 = np.zeros(diff1.shape, dtype=np.uint8)
        
        forground_mask0 = diff0 > 1.0
        forground_mask1 = diff1 > 1.0

        mask2 = np.zeros(diff1.shape, dtype=np.uint8)
        mask2[diff1 > 1.5] = 255
        mask2 = filter_largest_component(mask2)

        mask0[forground_mask0] = 255
        mask1[forground_mask1] = 255

        # deal mask1
        mask0 = filter_largest_component(mask0)
        mask1 = filter_largest_component(mask1)
        
        kernel = np.ones((20, 20), dtype=np.uint8)
        # mask1 = cv2.erode(mask1, kernel, iterations=1)
        
        #plt.imshow(erode_mask)
        #plt.show()

        mask = mask0 + mask1
        mask = filter_largest_component(mask)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask, mask2

    def get_diffs(self, img, bk):
        h, w = img.shape[0:2]

        diff_rgb = get_rgb_diff_reduced(img, bk, 0.2)

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

        diff0, diff1, diff2, diff3 = diffs

        img_g0 = cv2.GaussianBlur(img, (51, 51), 0)
        bk_g0 = cv2.GaussianBlur(bk, (51, 51), 0)

        img_g1 = cv2.GaussianBlur(img, (151, 151), 0)
        bk_g1 = cv2.GaussianBlur(bk, (151, 151), 0)

        diff_g0 = get_rgb_diff(img_g0, bk_g0, 1.0)
        diff_g1 = get_rgb_diff(img_g1, bk_g1, 1.0)

        # all_diffs = np.concatenate((diff_g0, ), axis=2)
        # all_diffs = np.concatenate((diff_g1, img_g1, bk_g1), axis=2)
        all_diffs = np.concatenate((diff_rgb[:, :, np.newaxis], diff0[:, :, np.newaxis], diff1[:, :, np.newaxis], diff2[:, :, np.newaxis], diff3[:, :, np.newaxis]), axis=2)
        
        
        # all_diffs = np.concatenate((diff_g0, diff_g1), axis=2)
        # all_diffs = np.concatenate((img_g0, bk_g0, img, bk, diff_rgb, diff0[:, :, np.newaxis], diff1[:, :, np.newaxis]), axis=2)
        # all_diffs = np.concatenate((img, bk, diff_rgb, diff0[:, :, np.newaxis], diff1[:, :, np.newaxis]), axis=2)
        # all_diffs = np.concatenate((diff_rgb, diff1[:, :, np.newaxis]), axis=2)
        # all_diffs = np.concatenate((diff_rgb, ), axis=2)
        # all_diffs = np.concatenate((diff_rgb, diff0[:, :, np.newaxis], diff1[:, :, np.newaxis]), axis=2)
        # all_diffs = np.concatenate((diff0[:, :, np.newaxis], diff1[:, :, np.newaxis]), axis=2)
        return all_diffs

def crop_out(img, mask):
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for c in range(img.shape[2]):
        new_img[:, :, c][mask > 0] = img[:, :, c][mask > 0]
    return new_img

def chop_img(img, patch_size, stride):
    h, w = img.shape[0:2]
    rbegin, cbegin = 0, 0
    rend, cend = rbegin + patch_size, cbegin + patch_size
    
    row_cnt, c_cnt = 0, 0
    patches = []

    while rbegin < h:
        rend = min(rbegin + patch_size, h)
        cbegin = 0
        while cbegin < w:
            cend = min(cbegin + patch_size, w)
            
            cur_patch = img[rbegin:rend, cbegin:cend, :]
            patches.append((cur_patch, (rbegin, rend, cbegin, cend)))
            cbegin += stride
            c_cnt += 1

        rbegin += stride
        row_cnt += 1
    
    return patches

def rebuild_mask(patches, img_info):
    pass

def get_trimap(mask):
    trimap = np.zeros(mask.shape, dtype=np.uint8)
    trimap[mask>0] = 128
    kernel = np.ones((10, 10), dtype=np.uint8)
    mask1 = cv2.erode(mask, kernel, iterations=20)
    trimap[mask1>0] = 255
    return trimap

def get_trimap2(mask, mask2):
    trimap = np.zeros(mask.shape, dtype=np.uint8)
    trimap[mask>0] = 128
    #trimap[mask2>0] = 255
    return trimap



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
    pos_mask = cv2.imread('pose_mask.png', 0)

    masker = DeepBkSubstractor()
    diffs = masker.get_diffs(img, bk)
    print(diffs.shape)

    """
    for i in range(diffs.shape[2]):  
        plt.imshow(diffs[:, :, i])
        plt.show()
    """

    # whiten
    for i in range(diffs.shape[2]):
        channel = diffs[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)

        diffs[:, :, i] = (channel - mean) / std

    
    """
    mask, mask2 = masker.get_mask(img, bk)
    kernel = np.ones((5, 5), dtype=np.uint8)
    neg_mask = cv2.dilate(mask, kernel, iterations=30)
    neg_mask[mask > 0] = 0
    """


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
        learned_mask = np.zeros(learned_mask_fgs.shape, dtype=np.uint8)
        learned_mask[learned_mask_fgs] = 255

        plt.imshow(score)
        plt.show()

        pos_mask = learned_mask
        #pos_mask_filled = filter_largest_component(pos_mask)

        #kernel = np.ones((15, 15), dtype=np.uint8)
        #neg_mask = cv2.dilate(pos_mask_filled, kernel, iterations=30)
        #neg_mask = neg_mask - pos_mask_filled

        #plt.imshow(neg_mask)
        #plt.show()

    """
    mask, mask2 = masker.get_mask(img, bk)

    trimap = get_trimap(mask)

    cv2.imwrite('test_img.png', img)
    cv2.imwrite('test_tri.png', trimap)
    """
    
    demo_img = crop_out(img, learned_mask)
    plt.imshow(demo_img[:, :, ::-1])
    plt.show()
    
    """
    trimap = mask.copy()
    trimap[mask > 0] = 128
    trimap[learned_mask > 0] = 255
    trimap[pos_mask > 0] = 255

    cv2.imwrite('test_tri_learned_sk.png', trimap)
    """

    




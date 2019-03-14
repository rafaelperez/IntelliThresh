import cv2
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from vgg import vgg16_bn_descriptor, ThreshNet
from utils import downsize_rgb, img_transform, crop_out

from human_parser.openpose_pytorch.api import OpenPoseForeGroundMarker
from trimap_generator import get_trimap_from_raw_mask_basic
# from matting.closed_form_matting.api import do_matting
from matting.grabcut.api import do_matting

class ThreshModel(object):
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def learn(self, features, labels):
        pass

    def infer(self, features):
        pass


class LinearThreshModel(ThreshModel):
    def __init__(self, feature_dim, learning_iters=5):
        super(LinearThreshModel, self).__init__(feature_dim)
        self.weights = None
        self.learning_iters = learning_iters
        
    def learn(self, features, labels):
        N = features.shape[0]
        feats = np.concatenate((features, np.ones((N, 1), dtype=np.float32)), axis=1)
        self.weights = np.linalg.inv(feats.transpose() @ feats) @ (feats.transpose() @ labels)
        # print(self.weights)
        
    def infer(self, features):
        assert self.weights is not None, 'weights are not learnt yet!'
        scores = features @ self.weights[0:self.feature_dim] + self.weights[self.feature_dim]
        return scores


class DeepThreshModel(ThreshModel):
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.learning_iters = 10
        self.batch_size = 1000
        self.epochs = 1

        self.model = ThreshNet(self.feature_dim, hidden_dim=1)
        self.device = torch.device('cuda:1')
        self.model.to(self.device)

        self.sample_feature_num = 1000000

    def learn(self, features, labels):

        self.model = ThreshNet(self.feature_dim, hidden_dim=1)
        self.device = torch.device('cuda:1')
        self.model.to(self.device)

        sample_ids = np.random.choice(a=features.shape[0], size=self.sample_feature_num)
        features = features[sample_ids, :]
        labels = labels[sample_ids, :]

        optimizer = optim.Adam(self.model.parameters())
        features = torch.from_numpy(features).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        dataset = TensorDataset(features, labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            print('epoch %d' % epoch)
            for feature, label in data_loader:
                pred = self.model(feature)
                loss = F.binary_cross_entropy(pred, label)
                loss.backward()
                optimizer.step()

                print('loss: %.4f' % loss.item())

    def infer(self, features):
        self.model.eval()
        features = torch.from_numpy(features).to(self.device)
        scores = self.model(features)
        scores = scores.detach().cpu().numpy()
        return scores


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

    def get_rgb_diff_scale_invariant(self, img, bg, factor):
        h, w = img.shape[:2]
        img_feature = downsize_rgb(img, factor)
        bg_feature = downsize_rgb(bg, factor)
        eps = 1.0
        bg_feature += eps
        diff = np.abs(img_feature - bg_feature)
        #diff = np.sum((img_feature - bg_feature)**2, axis=2)**0.5
        diff = diff / bg_feature
        diff = np.sum(diff**2, axis=2)**0.5
        diff = cv2.resize(diff, (w, h))
        return diff

    def get_diffs(self, img, bk):
        h, w = img.shape[0:2]

        diff_rgb = self.get_rgb_diff_scale_invariant(img, bk, self.rgb_downscale_factor)

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

        diffs.append(diff_rgb)
        diffs = np.stack(diffs, axis=2)
        # whiten
        means = np.mean(diffs, axis=(0, 1), keepdims=True)
        stds = np.std(diffs, axis=(0, 1), keepdims=True)
        diffs = (diffs - means) / stds
        return diffs

    def get_mask(self, img, bk, init_fg_mask):
        diffs = self.get_diffs(img, bk)
        h, w, feature_dim = diffs.shape

        # solve threashold model from disparity feature
        # we need an initial positive mask.
        # TODO: call OpenPose here
        # pos_mask = cv2.imread('pose_mask.png', 0)
        pos_mask = init_fg_mask
        learned_mask = np.zeros(img.shape[0:2], dtype=np.uint8)

        thresh_model = LinearThreshModel(feature_dim)
        # thresh_model = DeepThreshModel(feature_dim)

        for i in range(thresh_model.learning_iters):
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

            feats = np.concatenate((pos_feats, neg_feats), axis=0)

            labels = np.zeros((2*pos_num, 1), dtype=np.float32)
            labels[:pos_num, :] = 1

            # learn
            thresh_model.learn(feats, labels)

            # infer
            score = thresh_model.infer(diffs.reshape(h*w, feature_dim))
            score = score.reshape(h, w)

            learned_mask_fgs = score > 0.5
            learned_mask.fill(0)
            learned_mask[learned_mask_fgs] = 255

            #plt.imshow(learned_mask)
            #plt.show()

            pos_mask = learned_mask

        return learned_mask
        
        


if __name__ == '__main__':

    print('Setting up human parsers...')
    fg_marker = OpenPoseForeGroundMarker()
    print('Setting up image descriptor...')
    masker = DeepBkSubtractor()

    import test_list_chenglei_social_full as test_set
    
    test_set_tag = test_set.test_set_tag
    data_root = test_set.data_root
    test_img_list = test_set.test_img_list
    bg_dir = test_set.bg_dir

    attrs_of_interest = ['gray']

    output_dir = 'output'
    output_folder = str(datetime.datetime.now()).replace(' ', '_')
    output_folder_path = os.path.join(output_dir, output_folder)
    for attr in attrs_of_interest:
        output_folder_path += ('_' + attr)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    attrs_of_interest = set(attrs_of_interest)

    for item in test_img_list:
        frame_id, view_id, attrs = item
        if not attrs_of_interest.issubset(attrs):
            continue

        img_path = os.path.join(data_root, frame_id, view_id + '.png')
        bk_path = os.path.join(data_root, bg_dir, view_id + '.png')
        print('Doing {}...'.format(img_path))


        img = cv2.imread(img_path, 1)
        bk = cv2.imread(bk_path, 1)

        init_fg_mask = fg_marker.infer_fg(img)
    
        learned_mask = masker.get_mask(img, bk, init_fg_mask)

        trimap = get_trimap_from_raw_mask_basic(learned_mask)

        alpha = do_matting(img, trimap)

        learned_mask = alpha

        demo_img = crop_out(img, learned_mask)

        output_mask_name = '{}_{}_mask.png'.format(frame_id, view_id)
        output_demo_name = '{}_{}_demo.png'.format(frame_id, view_id)
        output_tri_name = '{}_{}_tri.png'.format(frame_id, view_id)

        output_mask_path = os.path.join(output_folder_path, output_mask_name)
        output_demo_path = os.path.join(output_folder_path, output_demo_name)
        output_tri_path = os.path.join(output_folder_path, output_tri_name)
        
        cv2.imwrite(output_mask_path, learned_mask)
        cv2.imwrite(output_demo_path, demo_img)
        cv2.imwrite(output_tri_path, trimap)
        print('Wrote to {}'.format(output_demo_path))




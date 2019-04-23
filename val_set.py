import os
import os.path as osp
import numpy as np

val_root = '/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/bks_val_imgs'
fg_dir = 'fgs'
bg_dir = 'bgs'
gt_dir = 'gts'
bg_path = osp.join(val_root, bg_dir)
gt_dir_path = osp.join(val_root, gt_dir)


def get_samples():
    fg_path = osp.join(val_root, fg_dir)
    img_paths = list()
    bk_paths = list()
    sample_ids = list()
    gt_paths = list()
    for root, dirs, files in os.walk(fg_path):
        for name in files:
            img_path = osp.join(root, name)

            bk_path = osp.join(bg_path, name)
            sample_sub_path = img_path[len(fg_path)+1:]
            sample_id = sample_sub_path.replace('/', '_').split('.')[0]

            gt_path = osp.join(gt_dir_path, sample_sub_path)

            img_paths.append(img_path)
            bk_paths.append(bk_path)
            sample_ids.append(sample_id)
            gt_paths.append(gt_path)

    return zip(img_paths, bk_paths, sample_ids, gt_paths)


def eval_mask(mask_pred, mask_gt):
    pos_mask = (mask_pred > 0)
    neg_mask = (mask_pred <= 0)
    
    tp_mask = (pos_mask) * mask_gt
    fp_mask = (pos_mask) * (1 - mask_gt)
    fn_mask = (neg_mask) * mask_gt

    tp_area = np.sum(tp_mask)
    fp_area = np.sum(fp_mask)
    gt_area = np.sum(mask_gt)

    precision = float(tp_area) / (tp_area + fp_area)
    recall = float(tp_area / gt_area)
    
    # draw error map
    H, W = mask_pred.shape
    err_map = np.where(mask_gt > 0, 255, 0).astype(np.uint8).reshape(H, W, 1)
    err_map = np.repeat(err_map, 3, axis=2)
    err_map[:, :, 0][fp_mask > 0] = 255 # blue for fp
    err_map[:, :, 2][fn_mask > 0] = 255 # red for fn

    return precision, recall, err_map



if __name__ == '__main__':
    get_samples()
import cv2
import numpy as np


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

    return new_mask


def filter_component_by_area(mask, min_area):
    output = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_16U)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    id_area_pair = [(stats[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, num_labels)]
    remain_ids = [pair[1] for pair in id_area_pair if pair[0] > min_area]
    
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    for remain_id in remain_ids:
        new_mask[labels == remain_id] = 255

    return new_mask


def filter_largest_component_and_fill(mask):
    new_mask = filter_largest_component(mask)

    max_comp_not_filled = new_mask

    new_mask_flood = new_mask.copy()
    cv2.floodFill(image=new_mask_flood, seedPoint=(0, 0), newVal=255, mask=None)
    new_mask = new_mask_flood - new_mask
    new_mask = 255 - new_mask

    return max_comp_not_filled, new_mask


def crop_out(img, mask):
    new_img = img.astype(np.float32)
    mask = mask / 255.0
    mask = mask[:, :, np.newaxis]
    new_img *= mask
    new_img = new_img.astype(np.uint8)
    return new_img


def get_trimap(mask):
    trimap = np.zeros(mask.shape, dtype=np.uint8)
    trimap[mask>0] = 128
    kernel = np.ones((10, 10), dtype=np.uint8)
    mask1 = cv2.erode(mask, kernel, iterations=20)
    trimap[mask1>0] = 255
    return trimap


def downsize_rgb(img, factor):
    h, w = img.shape[:2]
    nh, nw = int(h*factor), int(w*factor)
    img_rgb = cv2.resize(img, (nw, nh)).astype(np.float32)
    return img_rgb
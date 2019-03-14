import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import filter_largest_component_and_fill, filter_component_by_area

def do_matting(image, trimap, max_size=2000, confidence=50.0):
    assert image.shape[0:2] == trimap.shape[0:2]
    ori_h, ori_w = image.shape[0:2]
    scale = max_size / max(ori_h, ori_w)
    new_h, new_w = int(scale * ori_h), int(scale * ori_w)
    
    image = cv2.resize(image, (new_w, new_h))
    trimap = cv2.resize(trimap, (new_w, new_h))

    mask = np.zeros((new_h, new_w), np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask[(100 < trimap) * (trimap < 200)] = 3
    mask[trimap >= 200] = 1
    mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    
    
    mask = np.where((mask==2)|(mask==0), 0, 255).astype(np.uint8)
    
    mask = cv2.resize(mask, (ori_w, ori_h))
    mask = np.where(mask > 128, 255, 0).astype(np.uint8)
    
    # filter small holes
    mask_max_comp, mask_max_comp_filled = filter_largest_component_and_fill(mask)
    mask_inner_holes = mask_max_comp_filled - mask_max_comp
    mask_inner_big_holes = filter_component_by_area(mask_inner_holes, min_area=(0.01*ori_h)**2)
    mask = mask_max_comp_filled - mask_inner_big_holes
    
    return mask


if __name__ == '__main__':
    image = cv2.imread('image.png', cv2.IMREAD_COLOR)
    trimap = cv2.imread('trimap.png', cv2.IMREAD_GRAYSCALE)
    alpha = do_matting(image, trimap)
    cv2.imwrite('out2.png', alpha)
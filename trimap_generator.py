import cv2
import numpy as np
from utils import filter_largest_component, filter_component_by_area, filter_largest_component_and_fill

import matplotlib.pyplot as plt

def get_trimap_from_raw_mask_basic(mask):
    h, w = mask.shape

    filter_hole_dim = 0.01 * h
    min_hole_area = int(filter_hole_dim**2)

    ksize = 2*int(filter_hole_dim)
    kernel = np.ones((ksize, ksize),np.uint8)

    mask_max_comp, mask_max_comp_filled = filter_largest_component_and_fill(mask)
    mask_inner_holes = mask_max_comp_filled - mask_max_comp
    mask_inner_big_holes = filter_component_by_area(mask_inner_holes, min_area=min_hole_area)
    
    mask_unsure = mask_max_comp_filled - mask_inner_big_holes
    mask_fg = cv2.erode(mask_unsure, kernel, iterations=1)
    
    trimap = np.zeros((h, w), dtype=np.uint8)
    trimap[mask_unsure > 0] = 128
    trimap[mask_fg > 0] = 255

    return trimap


def get_trimap_from_fine_mask(mask):
    h, w = mask.shape

    ksize = 2*int(0.005 * h)
    kernel = np.ones((ksize, ksize),np.uint8)

    mask_unsure = cv2.dilate(mask, kernel, iterations=1)
    
    trimap = np.zeros((h, w), dtype=np.uint8)
    trimap[mask_unsure > 0] = 128
    trimap[mask > 0] = 255

    return trimap
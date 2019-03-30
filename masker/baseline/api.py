import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def postMorphongProcessing(img):
    kernel = np.ones((3, 3), dtype=np.uint8)
    erode = cv2.erode(img, kernel, iterations=5)
    dilate = cv2.dilate(erode, kernel, iterations=5)
    dilate = cv2.dilate(dilate, kernel, iterations=5)
    erode = cv2.erode(dilate, kernel, iterations=5)
    return erode

def filter_and_sort_contours(contours):
    min_len = 500
    remains = [c for c in contours if len(c) > min_len]
    remains.sort(key=lambda c: -len(c))
    return remains

def fill_regions(img, bkimg):
    h, w = img.shape[0:2]
    borders = np.zeros((h+2, w+2), np.uint8)
    bordersbk = np.zeros((h+2, w+2), np.uint8)
    borders[1:h+1, 1:w+1] = img
    bordersbk[1:h+1, 1:w+1] = bkimg
    
    ret, imgSholdbk = cv2.threshold(bordersbk, 100, 255, cv2.THRESH_BINARY)
    im2, contoursbk, hierarchy = cv2.findContours(imgSholdbk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contoursbk = filter_and_sort_contours(contoursbk)

    ret, imgShold = cv2.threshold(borders, 100, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(imgShold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = filter_and_sort_contours(contours)

    keep_index = 0

    for i in range(min(len(contours), len(contoursbk))):
        tmpregions0 = np.zeros(borders.shape, dtype=np.uint8)
        cv2.drawContours(tmpregions0, contoursbk, i, 255, cv2.FILLED, 8)
        tmpregions1 = np.zeros(borders.shape, dtype=np.uint8)
        cv2.drawContours(tmpregions1, contours, i, 255, cv2.FILLED, 8)

        mask0 = tmpregions0 == 255
        mask1 = tmpregions1 == 255
        mask_both = mask0 * mask1

        tmpnumval = np.sum(mask_both)
        tmpbasetotal = np.sum(mask0)
        tmpournumval = np.sum(mask1)

        tmpratio0 = tmpbasetotal*1.0 / tmpnumval
        tmpratio1 = tmpournumval*1.0 / tmpnumval

        if tmpnumval < 1e5 or tmpratio1 > 1.2:
            keep_index = i
            break
        else:
            if i+1 < len(contours):
                index = i+1

    final_mask = np.zeros(borders.shape, dtype=np.uint8)
    cv2.drawContours(final_mask, contours, keep_index, 255, cv2.FILLED, 8)
    return final_mask[1:1+h, 1:1+w]
    
def bkSubstract(img, bk):
    h, w = img.shape[0:2]
    bkh, bkw = bk.shape[0:2]

    assert h == bkh and w == bkw

    difimg = np.zeros((h, w), dtype=np.uint8)
    difimgbk = np.zeros((h, w), dtype=np.uint8)

    tmpbk = bk.astype(np.float32)
    tmpcap = img.astype(np.float32)

    mask = tmpbk[:, :, 0] < 33
    tmpbk[:, :, 0][mask] = 200
    difimgbk[mask] = 255

    tmpnewcol = np.sum((tmpbk - tmpcap)**2, axis=2)**0.5 * 20

    difsum = np.abs(tmpbk[:, :, 0] - tmpcap[:, :, 0])
    difimg[difsum > 15] = 255
    difimg[difsum <= 15] = 0

    difimg = postMorphongProcessing(difimg)
    difimgbk = postMorphongProcessing(difimgbk)

    final_mask = fill_regions(difimg, difimgbk)
    return final_mask


class FRLBaselineMasker(object):
    def __init__(self):
        pass
    
    def get_mask(self, img, bk):
        return bkSubstract(img, bk)


if __name__ == '__main__':
    data_root = '/home/mscv1/Desktop/FRL/ChengleiSocial/undistorted'
    bg_dir = '000000'
    img_dir = '020294'

    view_id = '400191'

    img = cv2.imread(os.path.join(data_root, img_dir, view_id + '.png'))
    bk = cv2.imread(os.path.join(data_root, bg_dir, view_id + '.png'))

    final_mask = bkSubstract(img, bk)

    plt.imshow(final_mask)
    plt.show()
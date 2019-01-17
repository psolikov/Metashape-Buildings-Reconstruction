import codecs, json
import numpy as np
import cv2

# Run this on //psdevscns/ps_storage/solikov/kazan_part/P084_1710_Kazan_III-IV_E-8

# path to local json: C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/selected_5
# path to local photos folder: C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/

# Function to get contour of a building
# It is required to have 3 files in provided location: image with the building, image with background markers, image with foreground markers
# format: 'name.JPG', 'name_bg.JPG', 'name_fg.JPG'
# example of args: location = '//psdevscns/ps_storage/solikov/Segmentation/Chunk/', filename = '2018_08_15_Naklon-Left_g201b20265_f004_0629'

# Home folder: '\\psdevscns\ps_storage\solikov\kazan_part2\Data'
def get_default_folder():
    return '//psdevscns/ps_storage/solikov/kazan_part2/Data/'

def get_contour(location = '//psdevscns/ps_storage/solikov/kazan_part2/Data/', filename):
    JPG = '.JPG'
    img = cv2.imread(location + filename + JPG)
    markers = cv2.imread(location + filename + '_fg' + JPG, 0)
    markers_bg = cv2.imread(location + filename + '_bg' + JPG, 0)
    labels = np.zeros_like(markers, dtype=np.int32)
    labels[markers == 255] = 1
    labels[markers_bg == 255] = 2
    out = cv2.watershed(img, labels)
    out_cp = out.copy()
    out_cp = out_cp.astype('uint8')
    out_cp[out_cp == 2] = 255
    out_cp[out_cp == -1] = 255
    out_cp[out_cp == 1] = 0
    im2, contours, hierarchy = cv2.findContours(out_cp, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
    cnt = None
    if (cv2.contourArea(contours[0]) < cv2.contourArea(contours[1])):
        cnt = contours[0]
    else:
        cnt = contours[1]
    return cnt

def approximate_contour(cnt):
    epsilon = 0.0002*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    return approx

def serrialize_cnt(cnt, path):
    b = cnt.tolist()
    with open(path, 'w') as outfile:
        json.dump(b, outfile, separators=(',', ':'), indent=4)
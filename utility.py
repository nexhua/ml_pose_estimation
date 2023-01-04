"""
Contains utility functions such as draw circle, line on image or show image and etc
"""

import cv2
import matplotlib.pyplot as plt
import configurations
import os
import numpy as np


class Util:
    def __init__(self):
        self.config = configurations.Configuration()

    def imread_from_id(self, id):
        """
        Provide image id(ex: 0001571514.png) and this function will return image from the original dataset
        """
        return cv2.imread(os.path.join(self.config.ORIGINAL_DATASET_BASE_DIR, id))

    def to_rgb(self, img):
        """
        Convert BGR color type to RGB
        """
        im = img.copy()
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def draw_circle(self, img, center_x, center_y, radius=5, bgr_color=(0, 0, 255), line_type=-1, shift=0):
        """
        Draw circle using cv2.circle method.
        """
        im = img.copy()
        return cv2.circle(im, (int(center_x), int(center_y)), radius, bgr_color, line_type, shift)

    def rescale(self, to_rescale, img_shape, target_size):
        scaled = to_rescale.reshape(-1, 2)
        org_y, org_x, org_c = img_shape
        target_y, target_x, target_c = target_size
        ratio_y = float(target_y) / float(org_y)
        ratio_x = float(target_x) / float(org_x)
        print()
        for item in scaled:
            x = item[0]
            y = item[1]
            new_x = int(round(ratio_x * x))
            new_y = int(round(ratio_y * y))
            item[0] = new_x
            item[1] = new_y
        return scaled

    def crop_square(self, img, size, interpolation=cv2.INTER_AREA):
        h, w = img.shape[:2]
        min_size = np.amin([h, w])

        # Centralize and crop
        crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2),
                       int(w/2-min_size/2):int(w/2+min_size/2)]
        resized = cv2.resize(crop_img, (size, size),
                             interpolation=interpolation)

        return resized

    def draw_keypoints(self, img, keypoints, radius=5, color=(0,0,255)):
        coords = keypoints.reshape(-1, 2)

        for coord in coords:
            x = coord[0]
            y = coord[1]
            img = self.draw_circle(img, x, y, radius, color)
        return img

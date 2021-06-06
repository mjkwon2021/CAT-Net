"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 7, 2020
"""
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
import torch
import cv2


class FantasticReality(AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str=None, is_auth_list: bool=False):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/FR_valid_list.txt"
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['FR']
        self._ori_path = os.path.join(self._root_path, "dataset/ColorRealImages")
        self._tamp_path = os.path.join(self._root_path, "dataset/ColorFakeImages")
        self._mask_path = os.path.join(self._root_path, "dataset/SegmentationFake")
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip() for t in f.readlines()]
        self._is_auth_list = is_auth_list

    def get_tamp(self, index):
        if not self._is_auth_list:
            # tampered image
            assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
            im_path = os.path.join(self._tamp_path, self.tamp_list[index])
            mask_path = os.path.join(self._mask_path, self.tamp_list[index].replace('.jpg', '.npz'))
            matrix = np.load(mask_path)
            mask = matrix['arr_0'].squeeze()
            mask[mask > 0] = 1
            # mask_view = mask.reshape(img_RGB.shape[0]//8, 8, img_RGB.shape[1]//8, 8).transpose(0, 2, 1, 3)
            # mask = np.average(mask_view, axis=(2, 3))  # down-sample
            return self._create_tensor(im_path, mask)
        else:
            # authentic image
            assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
            im_path = os.path.join(self._ori_path, self.tamp_list[index])
            return self._create_tensor(im_path, mask=None)

    def get_qtable(self, index):
        if self._is_auth_list:
            # tampered image
            assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
            im_path = os.path.join(self._tamp_path, self.tamp_list[index])
            DCT_coef, qtables = self._get_jpeg_info(im_path)
            Y_qtable = qtables[0]
            return Y_qtable
        else:
            # authentic image
            assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
            im_path = os.path.join(self._ori_path, self.tamp_list[index])
            DCT_coef, qtables = self._get_jpeg_info(im_path)
            Y_qtable = qtables[0]
            return Y_qtable

    # overrides
    def get_PIL_Image(self, index):
        file = Image.open(os.path.join(self._tamp_path, self.tamp_list[index])) if not self._is_auth_list else Image.open(os.path.join(self._ori_path, self.tamp_list[index]))
        return file

    def save_mask(self, index):
        mask_path = os.path.join(self._mask_path, self.tamp_list[index].replace('.jpg', '.npz'))
        matrix = np.load(mask_path)
        mask = matrix['arr_0'].squeeze()
        mask[mask > 0] = 255
        Image.fromarray(mask).save("Sp_FR_mask/"+self.tamp_list[index].replace('.jpg', '.png'))


if __name__ == '__main__':
    # Color Real Images (authentic)
    root = project_config.dataset_paths['FR']
    tamp_root = root / "dataset/ColorRealImages"
    imlist = []
    for file in os.listdir(tamp_root):
        if not file.lower().endswith(".jpg"):
            continue
        imlist.append(file)

    with open("FR_auth_list.txt", "w") as f:
        f.write('\n'.join(imlist) + '\n')
    print(len(imlist))  # 16592


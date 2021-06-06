"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020
"""
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image, ImageChops, ImageFilter
import torch
from pathlib import Path


class CASIA(AbstractDataset):
    """
    directory structure:
    CASIA (dataset_path["CASIA"] in project_config.py)
    ├── CASIA 1.0 dataset (download: https://github.com/CauchyComplete/casia1groundtruth)
    │   ├── Au (un-zip it)
    │   └── Modified TP (un-zip it)
    ├── CASIA 1.0 groundtruth
    │   ├── CM
    │   └── Sp
    ├── CASIA 2.0 (download: https://github.com/CauchyComplete/casia2groundtruth)
    │   ├── Au
    │   └── Tp
    └── CASIA 2 Groundtruth  => Run renaming script in the excel file located in the above repo.
                            Plus, rename "Tp_D_NRD_S_N_cha10002_cha10001_20094_gt3.png" to "..._gt.png"
    """
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str, read_from_jpeg=False):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/CASIA_list.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['CASIA']
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]
        self.read_from_jpeg = read_from_jpeg

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / (self.tamp_list[index][2] if self.read_from_jpeg else self.tamp_list[index][0])
        mask_path = self._root_path / self.tamp_list[index][1]
        if self.tamp_list[index][1] == 'None':
            mask = None
        else:
            mask = np.array(Image.open(mask_path).convert("L"))
            mask[mask > 0] = 1
        return self._create_tensor(tamp_path, mask)

    def get_qtable(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        if not str(tamp_path).lower().endswith('.jpg'):
            return None
        DCT_coef, qtables = self._get_jpeg_info(tamp_path)
        Y_qtable = qtables[0]
        return Y_qtable


if __name__ == '__main__':
    # CASIA2 has non-jpg files - we convert them here. You can choose original extension or jpeg when you test.
    root = project_config.dataset_paths['CASIA']

    imlist = []  # format: tamp.ext,mask.png,jpg_converted.jpg (if already .jpg, jpg_converted.jpg is same as tamp.ext)
    # CASIA2
    tamp_root = root / "CASIA 2.0/Tp"
    mask_root = root / "CASIA 2 Groundtruth"
    jpg_root = root / "CASIA 2.0/jpg"
    jpg_root.mkdir(exist_ok=True)
    for file in os.listdir(tamp_root):
        if file in ['Tp_D_NRD_S_B_ani20002_nat20042_02437.tif']:
            continue  # stupid file
        if not file.lower().endswith(".jpg"):
            if not file.lower().endswith(".tif"):
                print(file)
                continue
            # convert to jpg
            jpg_im = Image.open(tamp_root / file)
            jpg_im.save(jpg_root/(os.path.splitext(file)[0]+".jpg"), quality=100, subsampling=0)
            imlist.append(','.join([str(Path("CASIA 2.0/Tp") / file),
                                    str(Path("CASIA 2 Groundtruth") / (os.path.splitext(file)[0] + "_gt.png")),
                                    str(Path("CASIA 2.0/jpg") / (os.path.splitext(file)[0] + ".jpg"))]))
        else:
            imlist.append(','.join([str(Path("CASIA 2.0/Tp") / file),
                                str(Path("CASIA 2 Groundtruth") / (os.path.splitext(file)[0]+"_gt.png")),
                                str(Path("CASIA 2.0/Tp") / file)]))
        assert (mask_root/(os.path.splitext(file)[0]+"_gt.png")).is_file()
    print(len(imlist))  # 6042

    # mask validation
    new_imlist=[]
    for s in imlist:
        im, mask, _ = s.split(',')
        im_im = np.array(Image.open(root / im))
        mask_im = np.array(Image.open(root / mask))
        if im_im.shape[0] != mask_im.shape[0] or im_im.shape[1] != mask_im.shape[1]:
            print("Skip:", im, mask)
            continue
        new_imlist.append(s)

    with open("CASIA_list.txt", "w") as f:
        f.write('\n'.join(new_imlist)+'\n')
    print(len(new_imlist))  # 6025

    # CASIA2 authentic
    tamp_root = root / "CASIA 2.0/Au"
    jpg_root = root / "CASIA 2.0/jpg"
    jpg_root.mkdir(exist_ok=True)
    for file in os.listdir(tamp_root):
        if not file.lower().endswith(".jpg"):
            if not file.lower().endswith(".bmp"):
                print(file)
                continue
            # convert to jpg
            jpg_im = Image.open(tamp_root / file)
            jpg_im.save(jpg_root / (os.path.splitext(file)[0] + ".jpg"), quality=100, subsampling=0)
            imlist.append(','.join([str(Path("CASIA 2.0/Au") / file),
                                    'None',
                                    str(Path("CASIA 2.0/jpg") / (os.path.splitext(file)[0] + ".jpg"))]))
        else:
            imlist.append(','.join([str(Path("CASIA 2.0/Au") / file),
                                    'None',
                                    str(Path("CASIA 2.0/Au") / file)]))
    print(len(imlist))  # 6042

    with open("CASIA_v2_auth_list.txt", "w") as f:
        f.write('\n'.join(imlist)+'\n')



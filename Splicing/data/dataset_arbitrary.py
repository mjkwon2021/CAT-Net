"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
Sep 10, 2020
"""
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
import glob

class arbitrary(AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str, read_from_jpeg=False):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "input/*.png" (to be used in glob)
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self.tamp_list = list(glob.glob(tamp_list))
        self.read_from_jpeg = read_from_jpeg

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self.tamp_list[index]
        im = Image.open(tamp_path)
        if im.format != "JPEG":
            temp_jpg = f"____temp_{index:04d}.jpg"
            Image.open(tamp_path).convert('RGB').save(temp_jpg, quality=100, subsampling=0)
            tensors = self._create_tensor(temp_jpg, None)
            os.remove(temp_jpg)
        else:
            tensors = self._create_tensor(tamp_path, None)
        return tensors

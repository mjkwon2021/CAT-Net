"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
27 Jan 2021

Note: 'tampCOCO' includes four different custom datasets: cm_COCO, bcm_COCO, bcmc_COCO, sp_COCO
'bcm_COCO' is an alias of 'CM RAISE' in the paper.
'bcmc_COCO' is an alias of 'CM-JPEG RAISE' in the paper.
"""
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch


class tampCOCO(AbstractDataset):
    """
    directory structure
    tampCOCO (dataset_path["tampCOCO"] in project_config.py)
    ├── cm_images
    ├── cm_masks
    └── sp_images ...
    """
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/cm_COCO_train_list.txt"
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['tampCOCO']
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        mask_path = self._root_path / self.tamp_list[index][1]
        mask = np.array(Image.open(mask_path).convert('L'))
        mask[mask > 1] = 1
        return self._create_tensor(tamp_path, mask)

    def get_qtable(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        if not str(tamp_path).lower().endswith('.jpg'):
            return None
        DCT_coef, qtables = self._get_jpeg_info(str(tamp_path))
        Y_qtable = qtables[0]
        return Y_qtable

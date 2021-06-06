"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 12, 2020
"""
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
import torch


class IMD2020(AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str, read_from_jpeg=False):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/IMD_valid_list.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['IMD']
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]
        self.read_from_jpeg = read_from_jpeg

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / (self.tamp_list[index][3] if self.read_from_jpeg else self.tamp_list[index][0])
        mask_path = self._root_path / self.tamp_list[index][1]
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
    dir_list = []
    root = project_config.dataset_paths['IMD']
    jpg_root = root / "0jpg"
    jpg_root.mkdir(exist_ok=True)
    for dir, _, files in os.walk(root):
        if str(jpg_root) == dir:
            continue
        r = dir.replace(str(root),'')[1:]
        temp = []
        for file in files:
            if (file.lower().endswith(".jpg") or file.lower().endswith(".png")) and "mask" not in file and "orig" not in file:
                # format: tamp.ext, mask.png, ori.jpg, jpeg_compressed.jpg
                mask = os.path.splitext(file)[0] + "_mask.png"
                assert os.path.isfile(os.path.join(dir, mask))
                ori = [fi for fi in files if "orig" in fi][0]
                assert ".jpg" in ori.lower()
                if not file.lower().endswith(".jpg"):
                    # convert to jpg
                    jpg_im = Image.open(root / dir / file).convert('RGB')
                    jpg_im.save(jpg_root / (os.path.splitext(file)[0] + ".jpg"), quality=100, subsampling=0)
                    temp.append(','.join((os.path.join(r, file), os.path.join(r, mask), os.path.join(r, ori), os.path.join("0jpg",(os.path.splitext(file)[0] + ".jpg")))))
                else:
                    temp.append(','.join((os.path.join(r, file), os.path.join(r, mask), os.path.join(r, ori), os.path.join(r, file))))
        if len(temp) >= 1:
            dir_list.append(temp)

    g1_list = []
    for i, files in enumerate(dir_list):
        g1_list.extend(files)

    with open("IMD_list.txt", "w") as f:
        f.write('\n'.join(g1_list)+'\n')

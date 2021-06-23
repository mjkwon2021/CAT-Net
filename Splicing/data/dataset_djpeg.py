"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
Aug 4, 2020

This dataset is used for pretraining the DCT stream on double JPEG detection.
"""
import project_config
from Splicing.data import AbstractDataset


class Djpeg(AbstractDataset.AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/Djpeg_train.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['djpeg']
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        return self._create_tensor(tamp_path, int(self.tamp_list[index][1]))


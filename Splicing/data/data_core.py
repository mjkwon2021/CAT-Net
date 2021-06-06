"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 8, 2020
"""


import torch
from torch.utils.data import Dataset
import random

from Splicing.data.dataset_FantasticReality import FantasticReality
from Splicing.data.dataset_IMD2020 import IMD2020
from Splicing.data.dataset_CASIA import CASIA
# from Splicing.data.dataset_tampCOCO import tampCOCO
# from Splicing.data.dataset_NC16 import NC16
# from Splicing.data.dataset_Columbia import Columbia
# from Splicing.data.dataset_Carvalho import Carvalho
# from Splicing.data.dataset_compRAISE import compRAISE
# from Splicing.data.dataset_COVERAGE import COVERAGE
# from Splicing.data.dataset_CoMoFoD import CoMoFoD
# from Splicing.data.dataset_GRIP import GRIP
from Splicing.data.dataset_arbitrary import arbitrary


class SplicingDataset(Dataset):
    def __init__(self, crop_size, grid_crop, blocks=('RGB',), mode="train", DCT_channels=3, read_from_jpeg=False, class_weight=None):
        self.dataset_list = []
        if mode == "train":
            self.dataset_list.append(FantasticReality(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/FR_train_list.txt"))
            self.dataset_list.append(FantasticReality(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/FR_auth_train_list.txt", is_auth_list=True))
            self.dataset_list.append(IMD2020(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/IMD_train_list.txt", read_from_jpeg=read_from_jpeg))
            self.dataset_list.append(CASIA(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/CASIA_v2_train_list.txt", read_from_jpeg=read_from_jpeg))
            self.dataset_list.append(CASIA(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/CASIA_v2_auth_train_list.txt", read_from_jpeg=read_from_jpeg))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/cm_COCO_train_list.txt"))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/sp_COCO_train_list.txt"))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/bcm_COCO_train_list.txt"))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/bcmc_COCO_train_list.txt"))
            # self.dataset_list.append(compRAISE(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/compRAISE_train.txt"))
        elif mode == "valid":
            self.dataset_list.append(FantasticReality(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/FR_valid_list.txt"))
            self.dataset_list.append(FantasticReality(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/FR_auth_valid_list.txt", is_auth_list=True))
            self.dataset_list.append(IMD2020(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/IMD_valid_list.txt", read_from_jpeg=read_from_jpeg))
            self.dataset_list.append(CASIA(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/CASIA_v2_valid_list.txt", read_from_jpeg=read_from_jpeg))
            self.dataset_list.append(CASIA(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/CASIA_v2_auth_valid_list.txt", read_from_jpeg=read_from_jpeg))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/cm_COCO_valid_list.txt"))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/sp_COCO_valid_list.txt"))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/bcm_COCO_valid_list.txt"))
            # self.dataset_list.append(tampCOCO(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/bcmc_COCO_valid_list.txt"))
            # self.dataset_list.append(compRAISE(crop_size, grid_crop, blocks, DCT_channels, "Splicing/data/compRAISE_valid.txt"))
        elif mode == "arbitrary":
            self.dataset_list.append(arbitrary(crop_size, grid_crop, blocks, DCT_channels, "./input/*", read_from_jpeg=read_from_jpeg))
        else:
            raise KeyError("Invalid mode: " + mode)
        if class_weight is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weight)
        self.crop_size = crop_size
        self.grid_crip = grid_crop
        self.blocks = blocks
        self.mode = mode
        self.read_from_jpeg = read_from_jpeg
        self.smallest = 1869  # smallest dataset size (IMD:1869)

    def shuffle(self):
        for dataset in self.dataset_list:
            random.shuffle(dataset.tamp_list)

    def get_PIL_image(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_PIL_Image(index)

    def get_filename(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_tamp_name(index)

    def __len__(self):
        if self.mode == 'train':
            # class-balanced sampling
            return self.smallest * len(self.dataset_list)
        else:
            return sum([len(lst) for lst in self.dataset_list])

    def __getitem__(self, index):
        if self.mode == 'train':
            # class-balanced sampling
            if index < self.smallest * len(self.dataset_list):
                return self.dataset_list[index//self.smallest].get_tamp(index%self.smallest)
            else:
                raise ValueError("Something wrong.")
        else:
            it = 0
            while True:
                if index >= len(self.dataset_list[it]):
                    index -= len(self.dataset_list[it])
                    it += 1
                    continue
                return self.dataset_list[it].get_tamp(index)

    def get_info(self):
        s = ""
        for ds in self.dataset_list:
            s += (str(ds)+'('+str(len(ds))+') ')
        s += '\n'
        s += f"crop_size={self.crop_size}, grid_crop={self.grid_crip}, blocks={self.blocks}, mode={self.mode}, read_from_jpeg={self.read_from_jpeg}, class_weight={self.class_weights}\n"
        return s






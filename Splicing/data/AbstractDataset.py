"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 7, 2020
"""
from abc import ABC, abstractmethod
from PIL import Image, JpegImagePlugin
import numpy as np
import math
import jpegio  # See https://github.com/dwgoon/jpegio/blob/master/examples/jpegio_tutorial.ipynb
import torch_dct as dct
import torch
import random
import cv2


class AbstractDataset(ABC):
    YCbCr2RGB = torch.tensor([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]], dtype=torch.float64)

    def __init__(self, crop_size, grid_crop: bool, blocks: list, DCT_channels=3):
        """
        :param crop_size: (H, W) or None. H and W must be the multiple of 8 if grid_crop==True.
        :param grid_crop: T: crop within 8x8 grid. F: crop anywhere.
        :param blocks: 'RGB', 'rawRGB', 'DCTcoef', 'DCTvol, 'qtable'
        """
        self._crop_size = crop_size
        self._grid_crop = grid_crop
        for block in blocks:
            assert block in ['RGB', 'DCTcoef', 'DCTvol', 'qtable']
        if 'DCTcoef' in blocks or 'DCTvol' in blocks or 'rawRGB' in blocks:
            assert grid_crop
        if grid_crop and crop_size is not None:
            assert crop_size[0] % 8 == 0 and crop_size[1] % 8 == 0
        self._blocks = blocks
        self.tamp_list = None
        self.DCT_channels = DCT_channels

    def _get_jpeg_info(self, im_path):
        """
        :param im_path: JPEG image path
        :return: DCT_coef (Y,Cb,Cr), qtables (Y,Cb,Cr)
        """
        num_channels = self.DCT_channels
        jpeg = jpegio.read(str(im_path))

        # determine which axes to up-sample
        ci = jpeg.comp_info
        need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
        if num_channels == 3:
            if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
                need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
            if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
                need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
        else:
            need_scale[0][0] = 2
            need_scale[0][1] = 2

        # up-sample DCT coefficients to match image size
        DCT_coef = []
        for i in range(num_channels):
            r, c = jpeg.coef_arrays[i].shape
            coef_view = jpeg.coef_arrays[i].reshape(r//8, 8, c//8, 8).transpose(0, 2, 1, 3)
            # case 1: row scale (O) and col scale (O)
            if need_scale[i][0]==1 and need_scale[i][1]==1:
                out_arr = np.zeros((r * 2, c * 2))
                out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

            # case 2: row scale (O) and col scale (X)
            elif need_scale[i][0]==1 and need_scale[i][1]==2:
                out_arr = np.zeros((r * 2, c))
                DCT_coef.append(out_arr)
                out_view = out_arr.reshape(r*2//8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, :, :, :] = coef_view[:, :, :, :]
                out_view[1::2, :, :, :] = coef_view[:, :, :, :]

            # case 3: row scale (X) and col scale (O)
            elif need_scale[i][0]==2 and need_scale[i][1]==1:
                out_arr = np.zeros((r, c * 2))
                out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, ::2, :, :] = coef_view[:, :, :, :]
                out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

            # case 4: row scale (X) and col scale (X)
            elif need_scale[i][0]==2 and need_scale[i][1]==2:
                out_arr = np.zeros((r, c))
                out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, :, :, :] = coef_view[:, :, :, :]

            else:
                raise KeyError("Something wrong here.")

            DCT_coef.append(out_arr)

        # quantization tables
        qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float) for i in range(num_channels)]

        return DCT_coef, qtables

    def _create_tensor(self, im_path, mask):
        ignore_index = -1

        img_RGB = np.array(Image.open(im_path).convert("RGB"))

        h, w = img_RGB.shape[0], img_RGB.shape[1]

        if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks or 'qtable' in self._blocks:
            DCT_coef, qtables = self._get_jpeg_info(im_path)

        if mask is None:
            mask = np.zeros((h, w))

        if self._crop_size is None and self._grid_crop:
            crop_size = (-(-h//8) * 8, -(-w//8) * 8)  # smallest 8x8 grid crop that contains image
        elif self._crop_size is None and not self._grid_crop:
            crop_size = None  # use entire image! no crop, no pad, no DCTcoef or rawRGB
        else:
            crop_size = self._crop_size

        if crop_size is not None:
            # Pad if crop_size is larger than image size
            if h < crop_size[0] or w < crop_size[1]:
                # pad img_RGB
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
                temp[:img_RGB.shape[0], :img_RGB.shape[1], :] = img_RGB
                img_RGB = temp

                # pad mask
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), ignore_index)  # pad with ignore_index(-1)
                temp[:mask.shape[0], :mask.shape[1]] = mask
                mask = temp

                # pad DCT_coef
                if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks:
                    max_h = max(crop_size[0], max([DCT_coef[c].shape[0] for c in range(self.DCT_channels)]))
                    max_w = max(crop_size[1], max([DCT_coef[c].shape[1] for c in range(self.DCT_channels)]))
                    for i in range(self.DCT_channels):
                        temp = np.full((max_h, max_w), 0.0)  # pad with 0
                        temp[:DCT_coef[i].shape[0], :DCT_coef[i].shape[1]] = DCT_coef[i][:, :]
                        DCT_coef[i] = temp

            # Determine where to crop
            if self._grid_crop:
                s_r = (random.randint(0, max(h - crop_size[0], 0)) // 8) * 8
                s_c = (random.randint(0, max(w - crop_size[1], 0)) // 8) * 8
            else:
                s_r = random.randint(0, max(h - crop_size[0], 0))
                s_c = random.randint(0, max(w - crop_size[1], 0))

            # crop img_RGB
            img_RGB = img_RGB[s_r:s_r+crop_size[0], s_c:s_c+crop_size[1], :]

            # crop mask
            mask = mask[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1]]

            # crop DCT_coef
            if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks:
                for i in range(self.DCT_channels):
                    DCT_coef[i] = DCT_coef[i][s_r:s_r+crop_size[0], s_c:s_c+crop_size[1]]
                t_DCT_coef = torch.tensor(DCT_coef, dtype=torch.float)  # final (but used below)

        # handle 'RGB'
        if 'RGB' in self._blocks:
            t_RGB = (torch.tensor(img_RGB.transpose(2,0,1), dtype=torch.float)-127.5)/127.5  # final

        # handle 'DCTvol'
        if 'DCTvol' in self._blocks:
            T = 20
            t_DCT_vol = torch.zeros(size=(T+1, t_DCT_coef.shape[1], t_DCT_coef.shape[2]))
            t_DCT_vol[0] += (t_DCT_coef == 0).float().squeeze()
            for i in range(1, T):
                t_DCT_vol[i] += (t_DCT_coef == i).float().squeeze()
                t_DCT_vol[i] += (t_DCT_coef == -i).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef >= T).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef <= -T).float().squeeze()

        # create tensor
        img_block = []
        for i in range(len(self._blocks)):
            if self._blocks[i] == 'RGB':
                img_block.append(t_RGB)
            elif self._blocks[i] == 'DCTcoef':
                img_block.append(t_DCT_coef)
            elif self._blocks[i] == 'DCTvol':
                img_block.append(t_DCT_vol)
            elif self._blocks[i] == 'qtable':
                continue
            else:
                raise KeyError("We cannot reach here. Something is wrong.")

        # final tensor
        tensor = torch.cat(img_block)

        if 'qtable' not in self._blocks:
            return tensor, torch.tensor(mask, dtype=torch.long), 0
        else:
            return tensor, torch.tensor(mask, dtype=torch.long), torch.tensor(qtables[:self.DCT_channels], dtype=torch.float)

    @abstractmethod
    def get_tamp(self, index):
        pass

    def get_tamp_name(self, index):
        item = self.tamp_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item

    def get_PIL_Image(self, index):
        file = self.tamp_list[index][0]
        im = Image.fromarray(cv2.cvtColor(cv2.imread(str(self._root_path / file)), cv2.COLOR_BGR2RGB))
        return im

    def __len__(self):
        return len(self.tamp_list)


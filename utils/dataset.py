from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        # self.dic = np.array(7,4)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def preprocess_mask(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        torch.set_printoptions(edgeitems=10)
        return img_trans

    def RGB_2_class_idx(cls, mask_to_be_converted):
        mapping = {(0  , 255, 255): 0,     #urban_land
                   (255, 255, 0  ): 1,    #agriculture
                   (255, 0  , 255): 2,    #rangeland
                   (0  , 255, 0  ): 3,      #forest_land
                   (0  , 0  , 255): 4,      #water
                   (255, 255, 255):5,     #barren_land
                   (0  , 0  , 0  ):6}           #unknown

        # mapping = {(0.,1., 1.): 0,     #urban_land
        #         (1., 1., 0.): 1,    #agriculture
        #         (1., 0., 1.): 2,    #rangeland
        #         (0., 1., 0.): 3,      #forest_land
        #         (0., 0., 1.): 4,      #water
        #         (1.,1.,1.):5,     #barren_land
        #         (0.,0.,0.):6}           #unknown
        temp = np.array(mask_to_be_converted)
        temp = np.where(temp>=128, 255, 0)

        class_mask=torch.from_numpy(temp)
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.zeros(h, w, dtype=torch.long)
        for k in mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)
            mask_out[validx] = torch.tensor(mapping[k], dtype=torch.long)

        return mask_out

    def __getitem__(self, i):
        idx = self.ids[i]
        idx = idx[:len(idx) - 4]
        mask_file = glob(self.masks_dir + idx + '_mask' + '.png')
        img_file = glob(self.imgs_dir + idx + '_sat' + '.jpg')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess_mask(mask, self.scale)
        mask = self.RGB_2_class_idx(mask)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            # 'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

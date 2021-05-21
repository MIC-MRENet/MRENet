from os.path import splitext, split
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
# from PIL import Image
import SimpleITK as sitk
from skimage import transform
from sklearn import preprocessing
# import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        a = list(np.linspace(45, 71, num=27))
        self.ids = [str(int(i)) for i in a]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h, d = pil_img.shape
        newW, newH, newD = 80, 160, 160
        # newW, newH, newD = int(scale * w), int(scale * h), int(scale * d)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # if pil_img.max()==2:
        #     pil_img=(pil_img*127.5).astype(np.uint8)
        #     pil_img = transform.resize(pil_img, (newW, newH, newD), anti_aliasing=False)
        #     #ret1, pil_img = cv2.threshold(pil_img, 0.7, 1, cv2.THRESH_BINARY)
        if pil_img.max()==1:
            pil_img=(pil_img*255).astype(np.uint8)
            pil_img = transform.resize(pil_img, (int(newW), newH, newD), anti_aliasing=False)
            #ret1, pil_img = cv2.threshold(pil_img, 0.7, 1, cv2.THRESH_BINARY)

        else:
            
            pil_img = transform.resize(pil_img, (int(newW), newH, newD), anti_aliasing=False)

       


        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 3:
            img_nd = np.expand_dims(img_nd, axis=3)

        # HWC to CHW
        img_trans = img_nd.transpose((3, 0, 1, 2))

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + 'TMJ_'+idx + '.nii.gz')
        img_file = glob(self.imgs_dir + 'TMJ_'+idx + '.nii.gz')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0])
        mask = sitk.ReadImage(mask_file[0])
        mask = sitk.GetArrayFromImage(mask)
        img = sitk.ReadImage(img_file[0])
        img = sitk.GetArrayFromImage(img)
        # img = Image.open(img_file[0]).convert('L')

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

from os.path import splitext, split
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import SimpleITK as sitk
from skimage import transform
# import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

   
        a=list(np.linspace(0, 0, num=1))
        self.ids = [str(int(i)) for i in a]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img1, pil_img2, pil_img3, pil_img4, pil_img5, scale):
        w, h, d = 481,481,481
        newW, newH, newD = int(scale * w), int(scale * h), int(scale * d)
        assert newW > 0 and newH > 0, 'Scale is too small'
        if pil_img1.max()==1:
            pil_img=(pil_img1*255).astype(np.uint8)
            pil_img = transform.resize(pil_img, (newW, newH, newD), anti_aliasing=False)
            #ret1, pil_img = cv2.threshold(pil_img, 0.7, 1, cv2.THRESH_BINARY)
        else:
            # pil_img1 = transform.resize(pil_img1, (newW, newH, newD), anti_aliasing=False)
            pil_img1 = pil_img1/((pil_img1.max()-pil_img1.min())*0.5)
            # pil_img2 = transform.resize(pil_img2, (newW, newH, newD), anti_aliasing=False)
            pil_img2 = pil_img2/((pil_img2.max()-pil_img2.min())*0.5)
            # pil_img3 = transform.resize(pil_img3, (newW, newH, newD), anti_aliasing=False)
            pil_img3 = pil_img3/((pil_img3.max()-pil_img3.min())*0.5)
            # pil_img4 = transform.resize(pil_img4, (newW, newH, newD), anti_aliasing=False)
            pil_img4 = pil_img4/((pil_img4.max()-pil_img4.min())*0.5)
            pil_img5 = transform.resize(pil_img5, (newW, newH, newD), anti_aliasing=False)
            pil_img5 = pil_img5/((pil_img5.max()-pil_img5.min())*0.5)
            pil_img = np.vstack((np.expand_dims(pil_img1, axis=0), np.expand_dims(pil_img2, axis=0), np.expand_dims(pil_img3, axis=0), np.expand_dims(pil_img4, axis=0), np.expand_dims(pil_img5, axis=0)))
            pil_img = pil_img.transpose((1, 2, 3, 0))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 3:
            img_nd = np.expand_dims(img_nd, axis=3)

        # HWC to CHW
        img_trans = img_nd.transpose((3, 0, 1, 2))
        if img_trans.max() > 500:
            img_trans = img_trans / 32768
        if img_trans.max() > 2:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir+ 'TMJ_' + idx + '.nii.gz')
        img_file1 = glob(self.imgs_dir +'UNet-1/' +'TMJ_' +idx + '.nii.gz')
        img_file2 = glob(self.imgs_dir +'UNet-2/' +'TMJ_' +idx + '.nii.gz')
        img_file3 = glob(self.imgs_dir +'UNet-3/' +'TMJ_' +idx + '.nii.gz')
        img_file4 = glob(self.imgs_dir +'UNet-4/' +'TMJ_' +idx + '.nii.gz')
        img_file5 = glob('/data/zk/1/data/imgs/TMJ_imgs/'+'TMJ_' +idx + '.nii.gz')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        mask = sitk.ReadImage(mask_file[0])
        mask = sitk.GetArrayFromImage(mask)
        img1 = sitk.ReadImage(img_file1[0])
        img1 = sitk.GetArrayFromImage(img1)
        img2 = sitk.ReadImage(img_file2[0])
        img2 = sitk.GetArrayFromImage(img2)
        img3 = sitk.ReadImage(img_file3[0])
        img3 = sitk.GetArrayFromImage(img3)
        img4 = sitk.ReadImage(img_file4[0])
        img4 = sitk.GetArrayFromImage(img4)
        img5 = sitk.ReadImage(img_file5[0])
        img5 = sitk.GetArrayFromImage(img5)
        # img = Image.open(img_file[0]).convert('L')

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img1, img2, img3, img4, img5, self.scale)
        mask = self.preprocess(mask, 0, 0, 0,0, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

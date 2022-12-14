from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random

import matplotlib.pyplot as plt
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
        sample['left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['right'], (2, 0, 1))
        sample['right'] = torch.from_numpy(right) / 255.

        # disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)

        if 'pseudo_disp' in sample.keys():
            disp = sample['pseudo_disp']  # [H, W]
            sample['pseudo_disp'] = torch.from_numpy(disp)

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left', 'right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample

class RandomCrop(object):
    def __init__(self, img_height, img_width, probability,validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate
        self.probability = probability

    def __call__(self, sample):
        if np.random.random() < self.probability:
            ori_height, ori_width = sample['left'][0,0].shape[-2:]
            # ori_height, ori_width = sample['left'].shape[:2]

            print(self.img_height,ori_height,self.img_width,ori_width)
            assert self.img_height <= ori_height and self.img_width <= ori_width

            
            self.offset_x = np.random.randint(ori_width - self.img_width + 1)

            start_height = 0
            assert ori_height - start_height >= self.img_height

            self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)


            # for i in range(sample['left'].shape[1]):
            #     sample['left'][0,i] = self.crop_img(sample['left'][0,i])
            #     sample['right'][0,i] = self.crop_img(sample['right'][0,i])
            sample['left'] = self.crop_q(sample['left'])
            sample['right'] = self.crop_q(sample['right'])
            sample['disp'] = self.crop_img(sample['disp'])

            # for i in range(sample['left'].shape[1]):
            # sample['left'] = self.crop_img(sample['left'])
            # sample['right'] = self.crop_img(sample['right'])
            # sample['disp'] = self.crop_img(sample['disp'])
        return sample

    def crop_q(self, img):
        return img[:,:,:, self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]

    def crop_img(self, img):
        return img[:,self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]

               
class RandomCropResize(object):
    def __init__(self, img_height=0, img_width=0, probability=0.5,validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate
        self.probability = probability

    def __call__(self, sample):
        ori_height, ori_width = sample['left'][0,0].shape[:2]
        # ori_height, ori_width = sample['left'].shape[:2]

        assert self.img_height <= ori_height and self.img_width <= ori_width

        # x = np.random.rand()
        # print("!!!!!!!!!!!!!!!!!",x)
        if np.random.rand()<self.probability:
            # print("x",x)
            origin_sample = sample

            height = np.random.randint(224,260)
            width = np.random.randint(224,346)
            self.img_height = height
            self.img_width = width           
            
            self.offset_x = np.random.randint(ori_width - width + 1)

            start_height = 0
            assert ori_height - start_height >= height

            self.offset_y = np.random.randint(start_height, ori_height - height + 1)

            # for i in range(sample['left'].shape[1]):
            #     sample['left'][0,i] = self.crop_img(sample['left'][0,i])
            #     sample['right'][0,i] = self.crop_img(sample['right'][0,i])
            sample['left'] = self.crop_q(sample['left'])
            sample['right'] = self.crop_q(sample['right'])
            sample['disp'] = self.crop_img(sample['disp'])

            mask = sample['left'] == float('inf')
            if (mask == True).sum()/(height*width) > 0.8:
                sample == origin_sample

            # for i in range(sample['left'].shape[1]):
            # sample['left'] = self.crop_img(sample['left'])
            # sample['right'] = self.crop_img(sample['right'])
            # sample['disp'] = self.crop_img(sample['disp'])
        return sample

    def crop_q(self, img):
        return img[:,:,self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]

class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        

        if np.random.random() < 0.5:
            sample['left'] = np.copy(np.flipud(sample['left']))
            sample['right'] = np.copy(np.flipud(sample['right']))
            sample['disp'] = np.copy(np.flipud(sample['disp']))

        if np.random.random() < 0.5:
            sample['left'] = np.copy(np.fliplr(sample['left']))
            sample['right'] = np.copy(np.fliplr(sample['right']))
            sample['disp'] = np.copy(np.fliplr(sample['disp']))

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
        sample['right'] = Image.fromarray(sample['right'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['left'] = np.array(sample['left']).astype(np.float32)
        sample['right'] = np.array(sample['right']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)
            sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['left'] = F.adjust_gamma(sample['left'], gamma)
            sample['right'] = F.adjust_gamma(sample['right'], gamma)

        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['left'] = F.adjust_brightness(sample['left'], brightness)
            sample['right'] = F.adjust_brightness(sample['right'], brightness)

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['left'] = F.adjust_hue(sample['left'], hue)
            sample['right'] = F.adjust_hue(sample['right'], hue)

        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_saturation(sample['left'], saturation)
            sample['right'] = F.adjust_saturation(sample['right'], saturation)

        return sample


class RandomColor(object):

    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample

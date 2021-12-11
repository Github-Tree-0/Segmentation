import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from augment import strong_aug
import re
from utils.img_tool import *

# MAC_SIZE = 2200
# TRAIN_SHAPE = [(MAC_SIZE, MAC_SIZE), (MAC_SIZE, MAC_SIZE)]
# VALID_SHAPE = [(MAC_SIZE, MAC_SIZE), (MAC_SIZE, MAC_SIZE)]

class URISC(Dataset):
    def __init__(self, args, mode="train", transform=None):
        super(URISC, self).__init__()
        self.path = args.path
        self.mode = mode
        self.transform = transform
        self.crop_size = args.crop_size # 960
        self.augmentation = args.augmentation # False
        self.filenames = []
        # PATCHS
        tempdir1 = os.path.join(self.path, mode)
        for dirname in os.listdir(tempdir1):
            tempdir2=os.path.join(tempdir1, dirname)
            for filename in os.listdir(tempdir2):
                if filename[-4:].lower() == '.jpg':
                    self.filenames.append(os.path.join(tempdir2,filename))
        # self.filenames = [os.path.join(self.path, mode, filename) for filename in os.listdir(os.path.join(self.path, mode))]
        self.device = args.device
        self.mac_size = args.mac_size
        self.train_shape = [(self.mac_size, self.mac_size), (self.mac_size, self.mac_size)]
        self.val_shape = [(self.mac_size, self.mac_size), (self.mac_size, self.mac_size)]
        self.repeat = args.repeat
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        im = image_read(self.filenames[item])
        label_path = self.filenames[item][:-4] + '.png'
        # label_path = re.sub(r'(complex/)', r'\1label/', self.filenames[item])
        lab = image_read(label_path)
        
        if self.mode == "test":
            if self.transform is not None:
                im = self.transform(im)
            return self.filenames[item], im.unsqueeze(0)
        
        if self.mode == "val":
            if self.transform is not None:
                im = self.transform(im)
            image = im.transpose((2, 0, 1))
            label = lab.transpose((2, 0, 1))
            
            image = self.__image_transform(image)
            label = self.__mask_transform(label)
            
            # Cropping (Remove or replace it)
            h, w = image.shape[-2], image.shape[-1]
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            image = image[[0], y:y+self.crop_size, x:x+self.crop_size]
            label = label[[0], y:y+self.crop_size, x:x+self.crop_size]

            image = image.cuda(device=self.device)
            label = label.cuda(device=self.device)

            return image, label
            
        
        if self.mode == "train":
            if self.augmentation:
                p = strong_aug(p=.8, crop_size=self.train_shape[0])
                res = p(image=np.array(im),mask=np.array(lab))
                image = res['image'].transpose((2, 0, 1))
                label = res['mask'].transpose((2, 0, 1))
            else:
                image = im.transpose((2, 0, 1))
                label = lab.transpose((2, 0, 1))

            if self.transform is not None:
                # convert Image to torch, normalize pixel intensity from [0, 255] to [0, 1]
                image = self.transform(image)

            image = self.__image_transform(image)
            label = self.__mask_transform(label)

            if random.random() < 0.5:
                image = image * random.uniform(0.9, 1.1)

            # cropping
            h, w = image.shape[-2], image.shape[-1]
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            image = image[[0], y:y+self.crop_size, x:x+self.crop_size]
            label = label[[0], y:y+self.crop_size, x:x+self.crop_size]

            image = image.cuda(device=self.device)
            label = label.cuda(device=self.device)

            return image, label

    def __mask_transform(self, mask):
        mask = torch.from_numpy(np.array(mask)).float()
        mask[mask == 255] = 1.0
        return mask
    
    def __image_transform(self, images):
        images = torch.from_numpy(np.array(images)).float()
        return images
        
    def __len__(self):
        return len(self.filenames)
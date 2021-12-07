import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from augment import strong_aug
from utils.img_tool import *

TRAIN_SHAPE = [(1024, 1024), (1024, 1024)]
VALID_SHAPE = [(1024, 1024), (1024, 1024)]

class URISC(Dataset):
    def __init__(self, path, mode="train", transform=None, crop_size=960, augmentation=False):
        super(URISC, self).__init__()
        self.path = path
        self.mode = mode
        self.transform = transform
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.filenames = [os.path.join(path, mode, filename) for filename in os.listdir(os.path.join(path, mode))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item, count=16):
        im = image_read(self.filenames[item])
        if self.mode == "test":
            if self.transform is not None:
                im = self.transform(im)
            return self.filenames[item], im.unsqueeze(0)
        label_path = self.filenames[item].replace(self.mode, "label")
        lab = image_read(label_path)
        if self.mode == "val":
            if self.transform is not None:
                im = self.transform(im)
            return im.unsqueeze(0), self.__mask_transform(lab).unsqueeze(0)

        if self.augmentation:
            image, label = [], []
            for _ in range(count):
                p = strong_aug(p=.8, crop_size = TRAIN_SHAPE[0])
                res = p(image=np.array(im),mask=np.array(lab))
                image.append(res['image'].transpose((2, 0, 1)))
                label.append(res['mask'].transpose((2, 0, 1)))
            image = np.array(image);label=np.array(label)

        if self.transform is not None:
            # convert Image to torch, normalize pixel intensity from [0, 255] to [0, 1]
            image = self.transform(image)
        label = self.__mask_transform(label)

        if random.random() < 0.5:
            image = image * random.uniform(0.9, 1.1)

        # cropping
        h, w = image.shape[-2], image.shape[-1]
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        image = image[..., y:y+self.crop_size, x:x+self.crop_size]
        label = label[..., y:y+self.crop_size, x:x+self.crop_size]
        return image, label

    def __mask_transform(self, mask):
        mask = torch.from_numpy(np.array(mask)).float()
        mask[mask == 255] = 1.0
        return mask

    def __len__(self):
        return len(self.filenames)
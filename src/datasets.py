import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from augment import strong_aug

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

    def __getitem__(self, item):
        image = Image.open(self.filenames[item])
        if self.mode == "test":
            if self.transform is not None:
                image = self.transform(image)
            return self.filenames[item], image
        label_path = self.filenames[item].replace(self.mode, "labels").replace("png", "tiff")
        label = Image.open(label_path)
        if self.mode == "val":
            if self.transform is not None:
                image = self.transform(image)
            return image, self.__mask_transform(label)

        if self.augmentation:
            p = strong_aug(p=.8, crop_size = TRAIN_SHAPE[0])
            image, label = p(image,label)

        if self.transform is not None:
            # convert Image to torch, normalize pixel intensity from [0, 255] to [0, 1]
            image = self.transform(image)
        label = self.__mask_transform(label)

        if random.random() < 0.5:
            image = image * random.uniform(0.9, 1.1)

        # cropping
        h, w = image.shape[1], image.shape[2]
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        image = image[:, y:y+self.crop_size, x:x+self.crop_size]
        label = label[:, y:y+self.crop_size, x:x+self.crop_size]
        return image, label

    def __mask_transform(self, mask):
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
        mask[mask == 255] = 1.0
        return mask

    def __len__(self):
        return len(self.filenames)
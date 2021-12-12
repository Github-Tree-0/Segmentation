#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module is used for data augmentation. The package albumentation claims
# to work with pytorch, tensorflow, etc.

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomResizedCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, GaussNoise,
    RGBShift, RandomRain, RandomSnow, RandomShadow, RandomFog, ElasticTransform
)

def strong_aug(p=0.5, crop_size=(512, 512)):
    return Compose([
        RandomResizedCrop(crop_size[0], crop_size[1], scale=(0.3, 1.0), ratio=(0.75, 1.3), interpolation=4, p=0.8),
        RandomRotate90(),
        Flip(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=180, p=0.8),
        OneOf([
            OpticalDistortion(p=0.5),
            GridDistortion(p=0.5),
            IAAPiecewiseAffine(p=0.5),
            ElasticTransform(p=0.5),
        ], p=0.3)
    ], p=p)

import os
import numpy as np

import torch
import torch.utils.data as data

from .datasets import URISC

def Dataloader(args, transform=None):
    train_set = URISC(args, mode='train', transform=transform)
    val_set = URISC(args, mode='val')
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch, num_workers=args.workers, pin_memory=False, shuffle=False)
    
    return train_loader, val_loader

def test_Dataloader(args):
    test_set = URISC(args, mode='test')
    test_loader = torch.utils.data.Dataloader(test_set, batch_size=args.test_batch, num_workers=args.workers, pin_memory=False, shuffle=False)
    
    return test_loader
    
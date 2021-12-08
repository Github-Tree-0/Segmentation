import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import loss
from datasets import URISC
from models import U_Net
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 2000

writer = SummaryWriter('../log/')
save_path = '../checkpoints/'
path = '../data/complex/'
train_data = URISC(path, mode="train",crop_size=572, augmentation=True,transform=torch.Tensor)
val_data = URISC(path, mode='val')
model = U_Net.U_Net(output_ch=1)
model = model.cuda(device=0)
criterion = loss.dice_loss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)
iters = len(train_data)
min_loss_val = 1e9
for epoch in tqdm(range(EPOCHS)):
    cr_loss = 0
    for it in range(iters):
        inp, gt = train_data[it]
        inp = inp.cuda(device=0);gt=gt.cuda(device=0)
        optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        # TODO: Adapt input to U-net.
        print(inp.shape,gt.shape)
        pred = model(inp)
        gt_cropped=transforms.CenterCrop(pred.detach().shape)(gt)
        loss = criterion(pred, gt_cropped)
        loss.backward()
        optimizer.step()
        cr_loss += loss.detach()
    writer.add_scalar('loss',cr_loss,epoch)
    if epoch % 30 == 0: # validation
        cr_val = validation()
        if (cr_val < min_loss_val):
            min_loss_val = cr_val
            torch.save(model.state_dict(),save_path+str(epoch))

def validation():
    cr_loss = 0
    with torch.no_grad():
        for it in range(len(val_data)):
            inp, gt = val_data[it]
            pred = model(inp)
            loss = criterion(pred, gt)
            cr_loss += loss
    return cr_loss

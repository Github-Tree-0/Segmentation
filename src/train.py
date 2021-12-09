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
from Options import options

# EPOCHS = 2000
def train(args):
    writer = SummaryWriter(args.log_dir) # '../log/'
    save_path = args.save_path # '../checkpoints/'
    path = args.path # '../data/complex/'
    train_data = URISC(path, mode='train', augmentation=True)
    val_data = URISC(path, mode='val')
    model = U_Net.U_Net(output_ch=1)
    model = model.cuda(device=args.device) # 0
    criterion = loss.dice_loss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr) # 1e-4
    iters = len(train_data)
    min_loss_val = args.min_loss_val # 1e9
    epochs = args.epochs # 2000
    
    for epoch in tqdm(range(epochs)):
        
        cr_loss = 0
        for it in range(iters):
            inp, gt = train_data[it]
            optimizer.zero_grad()
            # import ipdb;ipdb.set_trace()
            # TODO: Adapt input to U-net.
            pred = model(inp)
            gt_cropped=transforms.CenterCrop(pred.detach().shape[-2:])(gt)
            loss = criterion(pred, gt_cropped)
            loss.backward()
            optimizer.step()
            cr_loss += loss.item()
            
        writer.add_scalar('loss', cr_loss, epoch)
        
        save_epoch = args.save_epoch
        
        if epoch % save_epoch == 0: # validation
            cr_val = 0
            with torch.no_grad():
                for it in range(len(val_data)):
                    inp,  criterion(pred, gt)
                    cr_val += loss.item()
            if (cr_val < min_loss_val):
                min_loss_val = cr_val
                torch.sgt = val_data[it]
                    pred = model(inp)
                    loss =ave(model.state_dict(),save_path+str(epoch))
                
# EPOCHS = 2000

# writer = SummaryWriter('../log/')
# save_path = '../checkpoints/'
# path = '../data/complex/'
# train_data = URISC(path, mode="train",crop_size=2012, augmentation=True,transform=torch.Tensor)
# val_data = URISC(path, mode='val')
# model = U_Net.U_Net(img_ch=1,output_ch=1)
# model = model.cuda(device=0)
# criterion = loss.dice_loss()
# optimizer = optim.Adam(model.parameters(),lr=1e-4)
# iters = len(train_data)
# min_loss_val = 1e9
# for epoch in tqdm(range(EPOCHS)):
#     cr_loss = 0
#     for it in range(iters):
#         inp, gt = train_data[it]
#         inp = inp.cuda(device=0);gt=gt.cuda(device=0)
#         # This should be included in the file.
#         optimizer.zero_grad()
#         # import ipdb;ipdb.set_trace()
#         # TODO: Adapt input to U-net.
#         print(inp.shape,gt.shape)
#         pred = model(inp)
#         gt_cropped=transforms.CenterCrop(pred.detach().shape[-2:])(gt)
#         loss = criterion(pred, gt_cropped)
#         loss.backward()
#         optimizer.step()
#         cr_loss += loss.detach()
#     writer.add_scalar('loss',cr_loss,epoch)
#     if epoch % 30 == 0: # validation
#         cr_val = validation()
#         if (cr_val < min_loss_val):
#             min_loss_val = cr_val
#             torch.save(model.state_dict(),save_path+str(epoch))

# def validation():
#     cr_loss = 0
#     with torch.no_grad():
#         for it in range(len(val_data)):
#             inp, gt = val_data[it]

if __name__ == "__main__":
    args = options.BaseOpts().parse()
    train(args)
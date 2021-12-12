import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import Loss
from dataloader.dataloader import Dataloader
from models import U_Net
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Options import options

# EPOCHS = 2000
def train(args):
    log_path = os.path.join(args.log_dir, args.save_name)
    while os.path.exists(log_path):
        log_path += '_'
    os.makedirs(log_path)
    writer = SummaryWriter(log_path) # '../log/'
    save_dir = os.path.join(args.save_dir, args.save_name) # '../checkpoints/'
    while os.path.exists(save_dir):
        save_dir += '_'
#     train_data = URISC(args, mode='train')
#     val_data = URISC(args, mode='val')
    train_loader, val_loader = Dataloader(args)
    model = U_Net.U_Net(output_ch=1)
    model = model.cuda(device=args.device) # 0
    criterion = Loss.dice_loss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr) # 1e-4
    min_loss_val = args.min_loss_val # 1e9
    epochs = args.epochs # 2000
    save_epoch = args.save_epoch

    checkpoint = None
    if args.use_save:
        weight_list = [ int(_[:-4]) for _ in os.listdir(args.load_dir)]
        latest = str(sorted(weight_list)[-1])+'.pth'
        print(f'Loaded {latest}')
        checkpoint = torch.load(os.path.join(args.load_dir, latest))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    iters_per_epoch = len(train_loader)
    print("iters per epoch:",iters_per_epoch)
    for epoch in range(epochs):
        if (checkpoint):
            epoch = checkpoint['epoch']
            min_loss_val = checkpoint['min_loss_val']
        print("EPOCH:",epoch)
        cr_loss = 0
        for i, sample in tqdm(enumerate(train_loader)):
            inp, gt = sample
            optimizer.zero_grad()
            # import ipdb;ipdb.set_trace()
            # TODO: Adapt input to U-net.
            pred = torch.sigmoid(model(inp))
            gt_cropped = transforms.CenterCrop(pred.detach().shape[-2:])(gt)
            loss = criterion(pred, gt_cropped)
            loss.backward()
            optimizer.step()
            cr_loss += loss.item()
            if i % 50 == 0:
                writer.add_scalar('batch_loss', cr_loss, epoch*iters_per_epoch+i)
                cr_loss = 0

        if epoch % save_epoch == 0: # validation
            cr_val = 0
            with torch.no_grad():
                for i, sample in enumerate(val_loader):
                    inp, gt = sample
                    pred = torch.sigmoid(model(inp))
                    gt_cropped=transforms.CenterCrop(pred.detach().shape[-2:])(gt)
                    loss = criterion(pred, gt_cropped)
                    cr_val += loss.item()
                inp_cropped = transforms.CenterCrop(pred.detach().shape[-2:])(inp)
                writer.add_images('validation batch', torch.stack((inp_cropped[0],pred[0],gt_cropped[0])), epoch,dataformats='NCHW')
                writer.add_scalar('Validation loss', cr_val,epoch)
            if (cr_val < min_loss_val):
                min_loss_val = cr_val
                print(f'Updated min_loss_val:{min_loss_val}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, '{}.pth'.format(epoch))
                torch.save({'epoch': epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'min_loss_val': cr_val}, save_path)
                
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
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = options.BaseOpts().parse()
    train(args)
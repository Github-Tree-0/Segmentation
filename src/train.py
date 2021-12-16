import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import Loss
from dataloader.dataloader import Dataloader
from models import U_Net,Link_U_Net,AttU_Net
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
    # model = U_Net.U_Net(output_ch=1,img_ch=1)
    # model = Link_U_Net.Link_U_Net(img_ch=1,output_ch=1)
    model = AttU_Net.AttU_Net(img_ch=1,output_ch=1)
    model = model.cuda(device=args.device) # 0
    # nel_crit = Loss.near_edge_loss(args.device)# Loss.dice_loss()
    bce_crit = Loss.BCELoss()
    focal_crit = Loss.focal_loss()
    val_criterion = Loss.fscore_loss()
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
            loss = focal_crit(pred, gt_cropped) + val_criterion(pred,gt_cropped) + bce_crit(pred,gt_cropped)
            loss.backward()
            optimizer.step()
            cr_loss += loss.item()
            if i % 50 == 49:
                writer.add_scalar('batch_loss', cr_loss, epoch*iters_per_epoch+i)
                cr_loss = 0

        if epoch % save_epoch == 0: # validation
            cr_val = 0
            with torch.no_grad():
                for i, sample in enumerate(val_loader):
                    inp, gt = sample
                    pred = (model(inp)>0) + 0.#torch.sigmoid(model(inp))
                    gt_cropped=transforms.CenterCrop(pred.detach().shape[-2:])(gt)
                    loss = val_criterion(pred, gt_cropped)
                    cr_val += loss.item()
                inp_cropped = transforms.CenterCrop(pred.detach().shape[-2:])(inp)
                writer.add_images('validation batch', torch.vstack((inp_cropped,pred,gt_cropped)), epoch,dataformats='NCHW')
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

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = options.BaseOpts().parse()
    train(args)
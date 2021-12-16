import numpy as np
import torch
import os
import cv2
from Options import options
from tqdm import tqdm
from dataloader.dataloader import test_Dataloader
from models import U_Net, AttU_Net, Link_U_Net

class cutter():
    def __init__(self, out_size, image_shape, pad):
        self.out_size = out_size
        self.shape = image_shape
        
        self.left_pad = int(pad / 2)
        self.right_pad = int(pad / 2 + (out_size-self.shape[1]) % out_size)
        self.up_pad = int(pad / 2)
        self.down_pad = int(pad / 2 + (out_size-self.shape[0]) % out_size)
        self.pad = int(pad / 2)
        
        self.h_iters = int(np.ceil(self.shape[0] / out_size))
        self.w_iters = int(np.ceil(self.shape[1] / out_size))
        
    def cut(self, image):
        image = np.pad(image, ((self.up_pad, self.down_pad), (self.left_pad, self.right_pad)), 'symmetric')
        imgs = []
        for i in range(self.h_iters):
            for j in range(self.w_iters):
                u = self.out_size * i + self.pad
                l = self.out_size * j + self.pad
                r = l + self.out_size
                d = u + self.out_size
                imgs.append(image[u-self.pad:d+self.pad, l-self.pad:r+self.pad])
        return imgs
        
    def stick(self, imgs):
        image = np.zeros(self.shape)
        for i in range(self.h_iters):
            for j in range(self.w_iters):
                index = j + i * self.w_iters
                u = self.out_size * i
                l = self.out_size * j
                r = min(l + self.out_size, self.shape[1])
                d = min(u + self.out_size, self.shape[0])
                # image[u:d, l:r] = imgs[index][self.pad:self.pad+d-u,self.pad:self.pad+r-l]
                image[u:d, l:r] = imgs[index][0:d-u,0:r-l]
        return image

def test(args):
    test_save_dir = args.test_save_dir # Modify this idiot
    test_loader = test_Dataloader(args) # Modify datasets.py
    os.makedirs(os.path.join(test_save_dir,args.save_name), exist_ok=True)
    # model = U_Net.U_Net(output_ch=1, img_ch=1)
    model = AttU_Net.AttU_Net(img_ch=1,output_ch=1)
    model = model.cuda(device=args.device)
    load_dir = os.path.join(args.load_dir,args.save_name)
    # import ipdb;ipdb.set_trace()
    weight_list = [ int(_[:-4]) for _ in os.listdir(load_dir)]
    latest = str(sorted(weight_list)[-1])+'.pth'
    print(f'Loading {latest}...')
    checkpoint = torch.load(os.path.join(load_dir, latest))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        for (name, image) in tqdm(test_loader):
            # import ipdb;ipdb.set_trace()
            image = image[0];name = name[0]
            image_cutter = cutter(args.out_size, image.shape, args.pad)
            imgs = image_cutter.cut(image)
            preds = []
            print('Testing image {}:'.format(name))
            for img in tqdm(imgs):
                inp = torch.from_numpy(img[np.newaxis][np.newaxis]).cuda(args.device)
                pred = model(inp).cpu().detach().numpy().squeeze()
                pred[pred<0] = 0
                pred[pred>0] = 1
                preds.append(pred)
            pred_image = image_cutter.stick(preds)*255
            save_path = os.path.join(test_save_dir,args.save_name,name)
            print(f'Saving {save_path} ...')
            cv2.imwrite(save_path, pred_image,[cv2.IMWRITE_PNG_COMPRESSION, 0])
            
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    args = options.BaseOpts().parse()
    test(args)
import argparse
import os
import torch

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--save_dir',      default='../checkpoints/')
        self.parser.add_argument('--path',          default='../data/')
        self.parser.add_argument('--log_dir',       default='../log/')
        self.parser.add_argument('--device',        default=0,      type=int)
        self.parser.add_argument('--lr',            default=1e-4,   type=float)
        self.parser.add_argument('--min_loss_val',  default=1e9,    type=float)
        self.parser.add_argument('--epochs',        default=2000,   type=int)
        self.parser.add_argument('--save_epoch',    default=30,     type=int)
        self.parser.add_argument('--use_save',      default=False,  action='store_true')
        
        #### Dataset ####
        self.parser.add_argument('--crop_size',     default=960,    type=int)
        self.parser.add_argument('--augmentation',  default=False,  action='store_true')
        self.parser.add_argument('--mac_size',      default=1024,   type=int)
        self.parser.add_argument('--repeat',        default=1,      type=int)
        
        #### Dataloader ####
        self.parser.add_argument('--batch_size',    default=1,      type=int)
        self.parser.add_argument('--val_batch',     default=1,      type=int)
        self.parser.add_argument('--workers',       default=12,     type=int)

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args

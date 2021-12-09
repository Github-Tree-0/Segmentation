import argparse
import os
import torch

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        #### Trainining Process ####
        self.parser.add_argument('--save_path',     default='../checkpoints/')
        self.parser.add_argument('--path',          default='../data/complex/')
        self.parser.add_argument('--device_num',    default=0,      type=int)
        self.parser.add_argument('--lr',            default=1e-4,   type=float)
        self.parser.add_argument('--min_loss_val',  default=1e9,    type=float)
        self.parser.add_argument('--epochs',        default=2000,   type=int)
        self.parser.add_argument('--save_epoch',    default=30,     type=int)
        
        #### Dataset ####
        

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
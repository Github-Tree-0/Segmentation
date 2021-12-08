import argparse
import os
import torch

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        #### Trainining Process ####
        self.parser.add_argument('--save_path',     default='../checkpoints/')

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
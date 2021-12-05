import torch
from torch.nn import BCELoss
from torch import nn


class focal_loss(nn.Module):
    def __init__(self, alpha=0.70, power=2):
        super().__init__()
        self.alpha = alpha
        self.power = power
        
    def forward(self, predicted, target):
        criterion = BCELoss(reduction="none")
        loss = criterion(predicted, target)
        loss = self.alpha * loss * (1 - target) * (predicted ** self.power) + (1 - self.alpha) * loss * target * ((1 - predicted) ** self.power)
        return loss.mean()
    

class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted, target):
        predicted = 1 - predicted
        target = 1 - target
        dice = 2 * torch.sum(predicted * target) / (torch.sum(predicted * predicted) + torch.sum(target * target))
        return 1 - dice


class fscore_loss(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
    
    def forward(self, predicted, target):
        predicted = 1 - predicted
        target = 1 - target

        tp = torch.sum(predicted * target)
        fp = torch.sum(predicted * (1 - target))
        fn = torch.sum((1 - predicted) * target)

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        fscore = (1 + self.beta * self.beta) * precision * recall / (self.beta * self.beta * precision + recall + epsilon)

        return 1 - fscore
    

class near_edge_loss(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        
    def forward(self, predicted, target):
        smoothing_kernel = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1,
                                     padding=(self.kernel_size-1) // 2, bias=False).cuda()
        smoothing_kernel.weight.data = torch.ones(self.kernel_size, self.kernel_size).float().unsqueeze(0).unsqueeze(0).cuda() / \
                                       float(self.kernel_size * self.kernel_size)
        smoothing_kernel.weight.requires_grad = False
        smoothed_target = smoothing_kernel(target)
        smoothed_target[smoothed_target < 0.98] = 0
        smoothed_target[smoothed_target != 0] = 1.0
        near_edge_target = torch.zeros_like(smoothed_target).cuda().float()
        near_edge_target[smoothed_target != target] = 1

        predicted = predicted * near_edge_target
        loss = binary_cross_entropy_loss(predicted, near_edge_target)

        return loss


class binary_cross_entropy_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted, target):
        criterion = BCELoss()
        loss = criterion(predicted, target)
        return loss


class weighted_bce_loss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        
    def forward(self, predicted, target):
        loss = - self.weight * (1 - target) * torch.log(1 - predicted) - (1 - self.weight) * target * torch.log(predicted)
        return loss.mean()
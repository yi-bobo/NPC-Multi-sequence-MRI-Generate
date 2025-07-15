
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, target):
        # outputs:分割结果
        # targets:真实结果
        outputs = outputs.reshape(-1)
        target = target.view(-1)
        
        # calculate intersection and union
        intersection = (outputs * target).sum()
        total = outputs.sum() + target.sum()
        
        # calculate dice coefficient
        dice = (2. * intersection + self.smooth) / (total + self.smooth)

        # dice loss is 1 - dice coefficient
        return 1 - dice
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.smooth = smooth
    def forward(self, preds, targets):
        p = torch.sigmoid(preds).view(-1)
        t = targets.view(-1)
        tp = (p * t).sum()
        fp = (p * (1-t)).sum()
        fn = ((1-p) * t).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tversky

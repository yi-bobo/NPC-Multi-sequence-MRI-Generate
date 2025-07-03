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
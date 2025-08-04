
import torch
import torch.nn as nn
from torchvision import models



class VGGLoss_3D(nn.Module):
    def __init__(self, pretrained_dir='/data1/weiyibo/NPC-MRI/Models/Pre_model/VGG/vgg19-dcbb9e9d.pth', device=None, multi_gpu=False):
        super(VGGLoss_3D, self).__init__()
        self.vggloss = VGGLoss(pretrained_dir, device)
        # if multi_gpu:
        #     self.vggloss = nn.DataParallel(self.vggloss)
        self.device = device

    def forward(self, x, y):
        _, _, n, _, _ = x.size()
        loss = 0
        for i in range(n):
            loss += self.vggloss(x[:, :, i, :, :].repeat(1, 3, 1, 1), y[:, :, i, :, :].repeat(1, 3, 1, 1))
        return loss / n
    
class VGGLoss(nn.Module):
    def __init__(self, pretrained_dir='/data1/weiyibo/NPC-MRI/Models/Pre_model/VGG/vgg19-dcbb9e9d.pth', device=None):
        super(VGGLoss, self).__init__()
        # self.vgg = Vgg19(pretrained_dir, device).cuda()
        self.vgg = Vgg19(pretrained_dir, device)
        
        # self.vgg = GetVGGModel(pretrained_dir).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, pretrained_dir='',device=None, requires_grad=False):
        super(Vgg19, self).__init__()
        self.device = device
        if pretrained_dir != '':
            model_vgg = models.vgg19(pretrained=False)
            model_vgg.load_state_dict(torch.load(pretrained_dir, map_location='cpu'))
            print('Successful download of pre-trained model from %s' % pretrained_dir)
            vgg_pretrained_features = model_vgg.features
        else:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from perceive_loss.ResNet_3D import resnet18
from collections import OrderedDict

class VGGLoss(nn.Module):
    def __init__(self, pretrained_dir=''):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(pretrained_dir).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, pretrained_dir='', requires_grad=False):
        super(Vgg19, self).__init__()
        if pretrained_dir != '':
            model_vgg = models.vgg19(pretrained=False)
            model_vgg.load_state_dict(torch.load(pretrained_dir, map_location='cuda:0'))
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
        # out = [h_relu1, h_relu2, h_relu3, h_relu4]
        return out


class VGGLoss_3D(nn.Module):
    def __init__(self, pretrained_dir=''):
        super(VGGLoss_3D, self).__init__()
        self.vggloss = VGGLoss(pretrained_dir)

    def forward(self, x, y):
        _, _, n, _, _ = x.size()
        loss = 0
        for i in range(n):
            loss += self.vggloss(x[:, :, i, :, :].repeat(1, 3, 1, 1), y[:, :, i, :, :].repeat(1, 3, 1, 1))
        return loss / n


class ResNet18_3D(torch.nn.Module):
    def __init__(self, pretrained_path='/home/zyw/.cache/torch/checkpoints/resnet_18_23dataset.pth', rank=0,
                 requires_grad=False):
        super(ResNet18_3D, self).__init__()
        ResNet18 = resnet18(shortcut_type='A', no_cuda=False).to(rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        state_dict = torch.load(pretrained_path, map_location)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        ResNet18.load_state_dict(new_state_dict)
        self.slice1 = nn.Sequential(*list(ResNet18.children())[:3])
        self.slice2 = nn.Sequential(*list(ResNet18.children())[3:5])
        self.slice3 = nn.Sequential(*list(ResNet18.children())[5:6])
        self.slice4 = nn.Sequential(*list(ResNet18.children())[6:7])
        self.slice5 = nn.Sequential(*list(ResNet18.children())[7:8])
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


class ResNet18_3D_Loss(nn.Module):
    def __init__(self, pretrained_path, rank=0):
        super(ResNet18_3D_Loss, self).__init__()
        self.ResNet18 = ResNet18_3D(pretrained_path, rank=rank).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x, y = self.ResNet18(x), self.ResNet18(y)
        loss = 0
        for i in range(len(x)):
            loss += self.weights[i] * self.criterion(x[i], y[i].detach())
        return loss


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


if __name__ == '__main__':
    a = ResNet18_3D(pretrained_path='/home/zyw/.cache/torch/checkpoints/resnet_18_23dataset.pth', rank=0)

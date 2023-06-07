from __future__ import print_function
from util.util import *
from collections import OrderedDict
import os
import torch
import numpy as np
import argparse


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 0.3081 + 0.1307) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def tensor2im3d(image_tensor):
    image_numpy = image_tensor[0].cpu().float().numpy()
    return image_numpy

def save_networks(opt, save_name, model, epoch):

    save_filename = '%s.pth' % (save_name)
    save_path = os.path.join(opt.model_results, save_filename)

    state = {
        'epoch': epoch + 1,
        'netG_state_dict': model.netG.state_dict(),
        'netD_state_dict': model.netD.state_dict(),
        'optimizer_G': model.optimizer_G.state_dict(),
        'optimizer_D': model.optimizer_D.state_dict()
    }

    torch.save(state, save_path)

def load_networks(opt, model):

    load_filename = '%s.pth' % (opt.load_name)
    load_path = os.path.join(opt.model_results, load_filename)

    # if isinstance(net, torch.nn.DataParallel):
    #     net = net.module
    print('loading the model from %s' % load_path)

    # state = torch.load(load_path)
    state = torch.load(load_path, map_location='cpu')
    pretrained_netG_dict = state['netG_state_dict']
    model_netG_dict = model.netG.state_dict()
    pretrained_netG_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_netG_dict}
    model_netG_dict.update(pretrained_netG_dict)
    model.netG.load_state_dict(model_netG_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中

    if opt.isTrain:
        pretrained_netD_dict = state['netD_state_dict']
        model_netD_dict = model.netD.state_dict()
        pretrained_netD_dict = {k: v for k, v in pretrained_netD_dict.items() if k in model_netD_dict}
        model_netD_dict.update(pretrained_netD_dict)
        model.netD.load_state_dict(model_netD_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中

        model.optimizer_G.load_state_dict(state['optimizer_G'])
        model.optimizer_D.load_state_dict(state['optimizer_D'])

    opt.epoch_count = state['epoch']
    print('Successfully loading the model from %s' % load_path)


def print_current_message(epoch, iters, dataset_size, lr, iter_acc, losses):
    message = '(epoch: %d, iters: %d/%d, lr: %.6f,  iter_acc:%.3f)' % (epoch, iters, dataset_size, lr, iter_acc)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message

def get_current_visuals(model):
    """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""

    real_A = tensor2im3d(model.real_A.data)
    fake_B = tensor2im3d(model.fake_B.data)
    real_B = tensor2im3d(model.real_B.data)
    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

def get_current_losses(model):
    """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
    errors_ret = OrderedDict()
    for name in model.loss_names:
        if isinstance(name, str):
            errors_ret[name] = float(getattr(model, 'loss_' + name))  # float(...) works for both scalar tensor and float number
    return errors_ret

def update_learning_rate(model, max_epochs, epoch, lr_max ):
    """Update learning rates for all the networks; called at the end of every epoch"""
    model.optimizer_G.param_groups[0]['lr'] = lr_max * (1 - epoch / max_epochs) ** 0.9
    model.optimizer_D.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def inverser_norm_ct(x, max_vax, max_min):
    x = (x+1) / 2
    x = (max_vax - max_min) * x - abs(max_min)
    return x

def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)
    if not opt.continue_train:
        with open(opt.file_name_txt, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

def normalization(data, min_value, max_value):
    if type(data) is not np.ndarray:
        data = data.numpy()
    nor_data = (data - min_value)/(max_value - min_value)
    last_data = (nor_data - 0.5)*2
    return last_data
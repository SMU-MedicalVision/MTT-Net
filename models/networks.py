import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from pytorch_wavelets import DWTForward, DWTInverse
from models.block.MSwin_transformer import MSwintransformer



###############################################################################
# Helper Functions
###############################################################################

bias_setting = False

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1) and (classname != 'ShareSepConv3d' or classname != 'ShareSepConv2d'):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            print('Norm initialized')
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, resolution, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'MTTNET':
        net = MTTNET(input_nc, output_nc, ngf, norm_layer=norm_layer, upsample_type='nearest', skip_connection=True, resolution=resolution)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayer3DDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayer3DDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'wave3DDiscriminator':  # more options
        net = wave3DDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ShareSepConv3d(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv3d, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv3d(x, expand_weight,
                        None, 1, (0, self.padding, self.padding), 1, inc)

class SmoothDilatedResidualBlock3d(nn.Module):
    def __init__(self, dim, dilation, norm_layer, activation=nn.ReLU(True)):
        super(SmoothDilatedResidualBlock3d, self).__init__()

        conv_block = [ShareSepConv3d(dilation[1] * 2 - 1),
                      nn.Conv3d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=bias_setting),
                      norm_layer(dim, affine=True),
                      activation,
                      ShareSepConv3d(dilation[1] * 2 - 1),
                      nn.Conv3d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=bias_setting),
                      norm_layer(dim, affine=True)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Pix2pix_3d(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm3d,
                 padding_type='zero', resblock_type='smoothdilated', upsample_type='nearest', skip_connection=True):
        assert (n_blocks >= 0)
        super(Pix2pix_3d, self).__init__()

        self.skip_connection = skip_connection
        activation = nn.ReLU(True)
        # activation = nn.Tanh()  # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # norm_layer is different to tf
        conv0 = [nn.Conv3d(input_nc, ngf, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=bias_setting), norm_layer(ngf),
                 activation]
        self.conv0 = nn.Sequential(*conv0)

        ### downsample
        mult = 1
        conv_down1 = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down1 = nn.Sequential(*conv_down1)

        mult = 2
        conv_down2 = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down2 = nn.Sequential(*conv_down2)

        mult = 4
        conv_down3 = [
            nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=(1, 2, 2), padding=1, bias=bias_setting),
            norm_layer(ngf * mult * 2), activation]
        self.conv_down3 = nn.Sequential(*conv_down3)


        ### upsample
        mult = 8
        convt_up3 = [nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up3 = nn.Sequential(*convt_up3)

        mult = 4
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv3 = [nn.Conv3d(in_channels, ngf * mult, kernel_size=3, stride=1, padding=1, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv3 = nn.Sequential(*decoder_conv3)

        mult = 4
        convt_up2 = [nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up2 = nn.Sequential(*convt_up2)

        mult = 2
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv2 = [nn.Conv3d(in_channels, ngf * mult, kernel_size=5, stride=1, padding=2, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv2 = nn.Sequential(*decoder_conv2)

        mult = 2
        convt_up1 = [nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up1 = nn.Sequential(*convt_up1)

        if skip_connection:
            in_channels = ngf * 2
        else:
            in_channels = ngf
        decoder_conv1 = [nn.Conv3d(in_channels, output_nc, kernel_size=3, stride=1, padding=1, bias=True), nn.Tanh()]
        self.decoder_conv1 = nn.Sequential(*decoder_conv1)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv_down1(x0)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)

        x4 = self.convt_up3(x3)
        if self.skip_connection:
            x4 = torch.cat((x4, x2), dim=1)  # batchsize*channnel*z*x*y
        x4 = self.decoder_conv3(x4)

        x5 = self.convt_up2(x4)
        if self.skip_connection:
            x5 = torch.cat((x5, x1), dim=1)
        x5 = self.decoder_conv2(x5)

        x6 = self.convt_up1(x5)
        if self.skip_connection:
            x6 = torch.cat((x6, x0), dim=1)
        out = self.decoder_conv1(x6)
        return out

class MTTNET(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm3d,
                 padding_type='zero', resblock_type='smoothdilated', upsample_type='nearest', skip_connection=True, resolution=[]):
        assert (n_blocks >= 0)
        super(MTTNET, self).__init__()

        self.skip_connection = skip_connection
        activation = nn.ReLU(True)
        # activation = nn.Tanh()  # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # norm_layer is different to tf
        conv0 = [nn.Conv3d(input_nc, ngf, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=bias_setting), norm_layer(ngf),
                 activation]
        self.conv0 = nn.Sequential(*conv0)

        ### downsample
        mult = 1
        conv_down1 = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down1 = nn.Sequential(*conv_down1)

        mult = 2
        conv_down2 = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down2 = nn.Sequential(*conv_down2)

        mult = 4
        conv_down3 = [
            nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=(1, 2, 2), padding=1, bias=bias_setting),
            norm_layer(ngf * mult * 2), activation]
        self.conv_down3 = nn.Sequential(*conv_down3)

        mult = 8
        head_swin = [8]
        aggregate1 = [nn.Conv3d(ngf * 2 * 2, ngf * 2, kernel_size=1, stride=1, bias=bias_setting),
                      norm_layer(ngf * 2), activation]
        self.aggregate1 = nn.Sequential(*aggregate1)
        self.mixall_1 = nn.Conv3d(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=1, stride=1)


        aggregate2 = [nn.Conv3d(ngf * 4 * 2, ngf * 4, kernel_size=1, stride=1, bias=bias_setting),
                      norm_layer(ngf * 4), activation]
        self.aggregate2 = nn.Sequential(*aggregate2)
        self.mixall_2 = nn.Conv3d(in_channels=ngf * 4, out_channels=ngf * 4,kernel_size=1, stride=1)



        aggregate3 = [nn.Conv3d(ngf * 8 * 2, ngf * 8, kernel_size=1, stride=1, bias=bias_setting),
                      norm_layer(ngf * 8), activation]
        self.aggregate3 = nn.Sequential(*aggregate3)
        self.mixall_3 = nn.Conv3d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=1, stride=1)


        self.MSwinBlock1 = MSwintransformer(ngf * mult, dilation=[1, 2, 2], activation=activation,
                                                  norm_layer=norm_layer, n_downsampling=n_downsampling,
                                                  last_window_size=[[2, 4, 4]], last_num_heads=head_swin, num_layer=1,
                                                  resolution=resolution)
        self.MSwinBlock2 = MSwintransformer(ngf * mult, dilation=[1, 2, 2], activation=activation,
                                              norm_layer=norm_layer, n_downsampling=n_downsampling,
                                              last_window_size=[[2, 4, 4]], last_num_heads=head_swin, num_layer=1,
                                              resolution=resolution)
        self.MSwinBlock3 = MSwintransformer(ngf * mult, dilation=[1, 2, 2], activation=activation,
                                              norm_layer=norm_layer, n_downsampling=n_downsampling,
                                              last_window_size=[[2, 4, 4]], last_num_heads=head_swin, num_layer=1,
                                              resolution=resolution)
        self.MSwinBlock4 = MSwintransformer(ngf * mult, dilation=[1, 2, 2], activation=activation,
                                              norm_layer=norm_layer, n_downsampling=n_downsampling,
                                              last_window_size=[[2, 4, 4]], last_num_heads=head_swin, num_layer=1,
                                              resolution=resolution)
        self.MSwinBlock5 = MSwintransformer(ngf * mult, dilation=[1, 2, 2], activation=activation,
                                              norm_layer=norm_layer, n_downsampling=n_downsampling,
                                              last_window_size=[[2, 4, 4]], last_num_heads=head_swin, num_layer=1,
                                              resolution=resolution)
        self.MSwinBlock6 = MSwintransformer(ngf * mult, dilation=[1, 2, 2], activation=activation,
                                              norm_layer=norm_layer, n_downsampling=n_downsampling,
                                              last_window_size=[[2, 4, 4]], last_num_heads=head_swin, num_layer=1,
                                              resolution=resolution)


        self.Branch_MSwin_1 = MSwintransformer(int(ngf * mult / 4), dilation=[1, 2, 2], activation=activation,
                                               norm_layer=norm_layer, n_downsampling=n_downsampling - 2,
                                               last_window_size=[[2, 4, 4]], last_num_heads=head_swin,num_layer=1,
                                               resolution=resolution)

        self.Branch_MSwin_2 = MSwintransformer(int(ngf * mult / 2), dilation=[1, 2, 2], activation=activation,
                                               norm_layer=norm_layer, n_downsampling=n_downsampling - 1,
                                               last_window_size=[[2, 4, 4]], last_num_heads=head_swin,num_layer=1,
                                               resolution=resolution)

        self.Branch_MSwin_3 = MSwintransformer(ngf * mult, dilation=[1, 2, 2], activation=activation,
                                                  norm_layer=norm_layer, n_downsampling=n_downsampling,
                                                  last_window_size=[[2, 4, 4]], last_num_heads=head_swin,num_layer=1,
                                                  resolution=resolution)


        ### upsample
        mult = 8
        convt_up3 = [nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up3 = nn.Sequential(*convt_up3)

        mult = 4
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv3 = [nn.Conv3d(in_channels, ngf * mult, kernel_size=3, stride=1, padding=1, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv3 = nn.Sequential(*decoder_conv3)

        mult = 4
        convt_up2 = [nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up2 = nn.Sequential(*convt_up2)

        mult = 2
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv2 = [nn.Conv3d(in_channels, ngf * mult, kernel_size=5, stride=1, padding=2, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv2 = nn.Sequential(*decoder_conv2)

        mult = 2
        convt_up1 = [nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up1 = nn.Sequential(*convt_up1)

        if skip_connection:
            in_channels = ngf * 2
        else:
            in_channels = ngf
        decoder_conv1 = [nn.Conv3d(in_channels, output_nc, kernel_size=3, stride=1, padding=1, bias=True), nn.Tanh()]
        self.decoder_conv1 = nn.Sequential(*decoder_conv1)


        self.patch_embed_1 = project(1, 32, [2, 2, 2], [2, 2, 2], nn.GELU, nn.LayerNorm)
        self.patch_embed_2 = project(1, 64, [2, 4, 4], [2, 4, 4], nn.GELU, nn.LayerNorm)
        self.patch_embed_3 = project(1, 128, [2, 8, 8], [2, 8, 8], nn.GELU, nn.LayerNorm)


        # 分类

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 2)


    def forward(self, input):
        x0 = self.conv0(input)

        x1 = self.conv_down1(x0)

        x11 = self.patch_embed_1(input)
        x11 = self.Branch_MSwin_1(x11)
        x1_concat = torch.cat([x11, x1], dim=1)
        x1 = self.aggregate1(x1_concat)
        x1 = self.mixall_1(x1) + x1

        x2 = self.conv_down2(x1)

        # patch embeddding 2
        x21 = self.patch_embed_2(input)
        x21 = self.Branch_MSwin_2(x21)
        x2_concat = torch.cat([x21, x2], dim=1)
        x2 = self.aggregate2(x2_concat)
        x2 = self.mixall_2(x2) + x2

        x3 = self.conv_down3(x2)

        # patch embeddding 3
        x31 = self.patch_embed_3(input)
        x31 = self.Branch_MSwin_3(x31)
        x3_concat = torch.cat([x31, x3], dim=1)
        x3 = self.aggregate3(x3_concat)
        x3 = self.mixall_3(x3) + x3

        x3 = self.MSwinBlock1(x3)
        x3 = self.MSwinBlock2(x3)
        x3 = self.MSwinBlock3(x3)
        x3 = self.MSwinBlock4(x3)
        x3 = self.MSwinBlock5(x3)
        x3 = self.MSwinBlock6(x3)

        x_class = self.avgpool(x3)
        x_class = x_class.view(x_class.size(0), -1)
        label_result = self.fc(x_class)
        label_result = F.softmax(label_result, dim=1)



        x4 = self.convt_up3(x3)
        if self.skip_connection:
            x4 = torch.cat((x4, x2), dim=1)  # batchsize*channnel*z*x*y
        x4 = self.decoder_conv3(x4)

        x5 = self.convt_up2(x4)
        if self.skip_connection:
            x5 = torch.cat((x5, x1), dim=1)
        x5 = self.decoder_conv2(x5)

        x6 = self.convt_up1(x5)
        if self.skip_connection:
            x6 = torch.cat((x6, x0), dim=1)
        out = self.decoder_conv1(x6)


        return out, label_result

class ResnetBlock3d(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock3d, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, norm_layer, activation, use_dropout):
        conv_block = []
        p = 1

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class project(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim//2, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv3d(out_dim//2, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim//2)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim//2, Ws, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        return x



##############################################################################
# basic  Discriminator
##############################################################################

class NLayer3DDiscriminator(nn.Module):

    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayer3DDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input,isDetach):
        """Standard forward."""
        return self.model(input)

class wave3DDiscriminator(nn.Module):

    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(wave3DDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.xfm = DWTForward(J=1, mode='zero', wave='haar')

        kw = 3
        padw = int(np.ceil((kw - 1) / 2))

        input_sequence = [
            nn.Conv3d(input_nc, input_nc, kernel_size=kw, stride=(1, 2, 2), padding=padw, bias=use_bias),
            norm_layer(input_nc),
            nn.LeakyReLU(0.2, True)
        ]

        self.input_model = nn.Sequential(*input_sequence)
        sequence = [nn.Conv3d(input_nc*5, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, isDetach=False):
        """Standard forward."""
        T, C, D, H, W = input.size()
        if isDetach:
            x_wave = torch.zeros([T, 5 * C, D, int(H / 2), int(W / 2)]).cuda()
        else:
            x_wave = torch.zeros([T, 5 * C, D, int(H / 2), int(W / 2)], requires_grad=True).cuda()
        for D_idx in range(D):
            Yl, Yh = self.xfm(input[:, :, D_idx, :, :])
            x_wave[:, 0:C, D_idx, :, :] = Yl
            x_wave[:, C:2 * C, D_idx, :, :] = Yh[0][:, :, 0, :, :]
            x_wave[:, 2 * C:3 * C, D_idx, :, :] = Yh[0][:, :, 1, :, :]
            x_wave[:, 3 * C:4 * C, D_idx, :, :] = Yh[0][:, :, 2, :, :]

        x_wave[:, 4 * C:5 * C, :, :, :] = self.input_model(input)
        return self.model(x_wave)





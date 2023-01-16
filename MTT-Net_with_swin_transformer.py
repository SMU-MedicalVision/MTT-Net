import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

class MTTNet_win2swin(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm3d, upsample_type='nearest', skip_connection=True, resolution=[]):
        assert (n_blocks >= 0)
        super(MTTNet_win2swin, self).__init__()

        bias_setting=False

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


        self.swinresBlock1 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock2 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock3 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock4 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock5 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock6 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8], resolution=resolution)


        self.patch_swin_1 = Swin_block(int(ngf * mult / 4), n_downsampling=n_downsampling - 2, last_window_size=[[2, 4, 4]], last_num_heads=head_swin, resolution=resolution)

        self.patch_swin_2 = Swin_block(int(ngf * mult / 2), n_downsampling=n_downsampling - 1, last_window_size=[[2, 4, 4]], last_num_heads=head_swin, resolution=resolution)

        self.patch_swin_3 = Swin_block(ngf * mult, n_downsampling=n_downsampling, last_window_size=[[2, 4, 4]], last_num_heads=head_swin, resolution=resolution)


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


        self.patch_embed_1 = project_patch(1, 32, [2, 2, 2], [2, 2, 2], nn.GELU, nn.LayerNorm)
        self.patch_embed_2 = project_patch(1, 64, [2, 4, 4], [2, 4, 4], nn.GELU, nn.LayerNorm)
        self.patch_embed_3 = project_patch(1, 128, [2, 8, 8], [2, 8, 8], nn.GELU, nn.LayerNorm)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 2)


    def forward(self, input):
        x0 = self.conv0(input)

        x1 = self.conv_down1(x0)

        x11 = self.patch_embed_1(input)
        x11 = self.patch_swin_1(x11)
        x1_concat = torch.cat([x11, x1], dim=1)
        x1 = self.aggregate1(x1_concat)
        x1 = self.mixall_1(x1) +x1


        x2 = self.conv_down2(x1)

        # patch embeddding 2
        x21 = self.patch_embed_2(input)
        x21 = self.patch_swin_2(x21)
        x2_concat = torch.cat([x21, x2], dim=1)
        x2 = self.aggregate2(x2_concat)
        x2 = self.mixall_2(x2) +x2

        x3 = self.conv_down3(x2)

        # patch embeddding 3
        x31 = self.patch_embed_3(input)
        x31 = self.patch_swin_3(x31)
        x3_concat = torch.cat([x31, x3], dim=1)
        x3 = self.aggregate3(x3_concat)
        x3 = self.mixall_3(x3) + x3

        x3 = self.swinresBlock1(x3)
        x3 = self.swinresBlock2(x3)
        x3 = self.swinresBlock3(x3)
        x3 = self.swinresBlock4(x3)
        x3 = self.swinresBlock5(x3)
        x3 = self.swinresBlock6(x3)

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

class Unet_swin(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm3d, upsample_type='nearest', skip_connection=True, resolution=[]):
        assert (n_blocks >= 0)
        super(Unet_swin, self).__init__()
        bias_setting = False
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
        self.swinresBlock1 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock2 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock3 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock4 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock5 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)
        self.swinresBlock6 = Swin_block(ngf * mult, n_downsampling = n_downsampling, last_window_size=[[2, 4, 4]],last_num_heads=[8],resolution=resolution)

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

        x3 = self.swinresBlock1(x3)
        x3 = self.swinresBlock2(x3)
        x3 = self.swinresBlock3(x3)
        x3 = self.swinresBlock4(x3)
        x3 = self.swinresBlock5(x3)
        x3 = self.swinresBlock6(x3)


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

class Swin_block(nn.Module):
    def __init__(self, dim, n_downsampling, last_window_size, last_num_heads, resolution):
        super(Swin_block, self).__init__()

        depths = [2]
        self.pos_drop = nn.Dropout(p=0.0)

        self.last_hidden_size = dim
        self.last_window_size = last_window_size
        self.last_num_heads = last_num_heads
        self.last_num_layers = len(last_num_heads)

        self.last_patch_embeddings = nn.Conv3d(in_channels=self.last_hidden_size, out_channels=self.last_hidden_size,
                                               kernel_size=1,
                                               stride=1)

        self.lastlayers = nn.ModuleList()
        for i_layer in range(self.last_num_layers):
            layer = BasicLayer(
                dim=self.last_hidden_size,
                input_resolution=(4, int(resolution[1] / (2 ** (n_downsampling))),
                    int(resolution[2] / (2 ** (n_downsampling)))),
                depth=depths[0],
                num_heads=last_num_heads[i_layer],
                window_size=last_window_size[i_layer],
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0,
                drop_path=[x.item() for x in torch.linspace(0, 0.05, sum(depths))],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False)
            self.lastlayers.append(layer)
        self.last_norm_layer = nn.LayerNorm(self.last_hidden_size)

    def forward(self, x):

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = self.last_patch_embeddings(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patch, hidden)
        x = self.pos_drop(x)

        for i in range(self.last_num_layers):
            layer = self.lastlayers[i]
            x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = self.last_norm_layer(x)

        x = x.permute(0, 2, 1)
        x = x.view(-1, self.last_hidden_size, Ws, Wh, Ww).contiguous()

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 use_checkpoint=False,
                 i_layer=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.i_layer = i_layer
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            if i_layer == 1 or i_layer == 2:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=1)
            else:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=0)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):

        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size[0])) * self.window_size[0]
        Hp = int(np.ceil(H / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(W / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
        s_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        h_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            if self.i_layer != 1 and self.i_layer != 2:
                Ws, Wh, Ww = S, (H + 1) // 2, (W + 1) // 2
            else:
                Ws, Wh, Ww = S // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) == tuple(self.window_size):
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = [0, 0, 0]
            # self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):

        B, L, C = x.shape
        S, H, W = self.input_resolution

        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - S % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class project_patch(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim//2, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv3d(out_dim//2, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
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
        x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        return x

def window_partition(x, window_size):
    B, S, H, W, C = x.shape

    x = x.view(B, S // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, S, H, W):

    B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, S // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x

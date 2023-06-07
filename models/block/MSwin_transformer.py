import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import init
import functools
import ml_collections
from timm.models.layers import DropPath, trunc_normal_

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

class MSwintransformer(nn.Module):
    def __init__(self, dim, dilation, norm_layer, n_downsampling, last_window_size, last_num_heads, num_layer, resolution, activation=nn.ReLU(True)):
        super(MSwintransformer, self).__init__()
        bias_setting =  False
        conv_block = [ShareSepConv3d(dilation[1] * 2 - 1),
                      nn.Conv3d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=bias_setting),
                      norm_layer(dim, affine=True),
                      activation,
                      ShareSepConv3d(dilation[1] * 2 - 1),
                      nn.Conv3d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=bias_setting),
                      norm_layer(dim, affine=True)]
        self.conv_block = nn.Sequential(*conv_block)

        depths = [2]
        self.pos_drop = nn.Dropout(p=0.0)

        self.last_hidden_size = dim
        self.change_dim = dim * 2
        self.last_window_size = last_window_size
        self.last_num_heads = last_num_heads
        self.last_num_layers = num_layer

        self.last_patch_embeddings = nn.Conv3d(in_channels=self.last_hidden_size, out_channels=self.last_hidden_size,
                                               kernel_size=1,
                                               stride=1)

        self.lastlayers = nn.ModuleList()
        for i_layer in range(self.last_num_layers):
            layer = BasicLayer(
                dim=self.last_hidden_size,
                input_resolution=(
                int(resolution[0] / 2), int(resolution[1] / (2 ** (n_downsampling))),
                int(resolution[2] / (2 ** (n_downsampling)))),
                depth=depths[0],
                num_heads=last_num_heads[0],
                window_size=last_window_size[0],
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0,
                drop_path=[x.item() for x in torch.linspace(0, 0.05, sum(depths))],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
                i_layer=i_layer)
            self.lastlayers.append(layer)
        self.last_norm_layer = nn.LayerNorm(self.last_hidden_size)




    def forward(self, x):
        x_orgin = x
        x_conv = self.conv_block(x)
        x = x + x_conv

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = self.last_patch_embeddings(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patch, hidden)
        x = self.pos_drop(x)

        x_shout = x
        for i in range(self.last_num_layers):
            layer = self.lastlayers[i]
            x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            x = self.last_norm_layer(x)
            x = x + x_shout
            x_shout = x

        x = x.permute(0, 2, 1)
        x = x.view(-1, self.last_hidden_size, Ws, Wh, Ww).contiguous()


        out = x + x_orgin + x_conv

        return out

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
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.i_layer = i_layer
        self.dim = dim

        '''第一层'''
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.block11 = nn.ModuleList([
            Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)])


        '''第二层'''

        self.dim_mix = nn.Linear(dim, dim)

        self.window_size_total = []

        self.window_size_total.append([self.window_size[0], self.window_size[1] // 2, self.window_size[2] * 2])
        self.window_size_total.append([self.window_size[0], self.window_size[1] * 2, self.window_size[2] // 2])

        self.shift_size_total = []
        self.shift_size_total.append(
            [self.window_size_total[0][0] // 2, self.window_size_total[0][1] // 2, self.window_size_total[0][2] // 2])
        self.shift_size_total.append(
            [self.window_size_total[1][0] // 2, self.window_size_total[1][1] // 2, self.window_size_total[1][2] // 2])

        self.dim_change = [0.5,0.5]

        # build blocks
        self.block1 = nn.ModuleList([
            Block(
                dim=int(dim*self.dim_change[0]),
                input_resolution=input_resolution,
                num_heads=int(num_heads*self.dim_change[0]),
                window_size=self.window_size_total[0],
                shift_size=self.shift_size_total[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)])

        self.block2 = nn.ModuleList([
            Block(
                dim=int(dim * self.dim_change[1]),
                input_resolution=input_resolution,
                num_heads=int(num_heads * self.dim_change[1]),
                window_size=self.window_size_total[1],
                shift_size=self.shift_size_total[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)])

        # patch merging layer
        if downsample is not None:
            if i_layer == 1 or i_layer == 2:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=1)
            else:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=0)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
        '''第一层'''
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
        for blk in self.block11:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        '''第二层'''
        x_total  = []
        for idx in range(2):
            window_size = self.window_size_total[idx]
            shift_size = self.shift_size_total[idx]
            if idx ==0:
                block = self.block1
                x_this = x[:, :, 0:int(self.dim * self.dim_change[0])]
            elif idx==1:
                block = self.block2
                x_this = x[:, :, int(self.dim * self.dim_change[0]):]

            # calculate attention mask for SW-MSA
            Sp = int(np.ceil(S / window_size[0])) * window_size[0]
            Hp = int(np.ceil(H / window_size[1])) * window_size[1]
            Wp = int(np.ceil(W / window_size[2])) * window_size[2]
            img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
            s_slices = (slice(0, -window_size[0]),
                        slice(-window_size[0], -shift_size[0]),
                        slice(-shift_size[0], None))
            h_slices = (slice(0, -window_size[1]),
                        slice(-window_size[1], -shift_size[1]),
                        slice(-shift_size[1], None))
            w_slices = (slice(0, -window_size[2]),
                        slice(-window_size[2], -shift_size[2]),
                        slice(-shift_size[2], None))
            cnt = 0
            for s in s_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, s, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1,
                                             window_size[0] * window_size[1] * window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            for blk in block:
                blk.H, blk.W = H, W
                x_this = blk(x_this, attn_mask)
            x_total.append(x_this)

        x = torch.cat([x_total[0],x_total[1]],dim=2)
        x = self.dim_mix(x)


        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            if self.i_layer != 1 and self.i_layer != 2:
                Ws, Wh, Ww = S, (H + 1) // 2, (W + 1) // 2
            else:
                Ws, Wh, Ww = S // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W

class Block(nn.Module):
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

    def __init__(self, dim, input_resolution, num_heads, window_size=7,  shift_size=0,
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
            self.shift_size = [0,0,0]
            #self.window_size = min(self.input_resolution)
        #assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

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
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1],-self.shift_size[2]), dims=(1, 2,3))
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
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
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

        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_coords[:, :, 0] *= 2 * (self.window_size[1] + self.window_size[2]) - 1

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
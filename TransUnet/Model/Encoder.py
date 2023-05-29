import math
import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Feedward(nn.Module):
    def __init__(self, dim, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size()[-1])
        return torch.matmul(attention, v)


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim, n_head) -> None:
        super().__init__()
        self.n_head = n_head
        self.attention = Attention(dim)
        self.multi_head = Rearrange(pattern='b n (n_head f) -> b n n_head f', n_head=n_head)

    def forward(self, x):
        # Input shape: [batch, n_patch, n_feature]
        x = self.multi_head(x)
        x = self.attention(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, n_heads, dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_heads * dim))
        self.beta = nn.Parameter(torch.zeros(n_heads * dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class Encoder_layer(nn.Module):
    def __init__(self, dim, dropout, n_head) -> None:
        super().__init__()
        self.norm = LayerNorm(n_head, dim)
        self.droupout = nn.Dropout(dropout)
        self.multi_attn = Multi_Head_Attention(dim, n_head)
        self.recover = Rearrange(pattern='b n n_head f -> b n (n_head f)', n_head=n_head)

    def forward(self, x):
        x_ = x
        x = self.multi_attn(x)
        x = self.norm(self.recover(x) + x_)
        x = self.droupout(x)
        x_ = x
        x = self.multi_attn(x)
        x = self.norm(self.recover(x) + x_)
        x = self.droupout(x)
        return x


class Trans_Encoder(nn.Module):
    def __init__(self, emb_height, emb_width, args) -> None:
        super().__init__()
        assert emb_height % args.patch_height == 0 and emb_width % args.patch_width == 0
        # Patch Embedding
        self.patch_emb = nn.Sequential(
            Rearrange(pattern='b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=args.patch_height, p2=args.patch_width),
            nn.Linear(args.patch_height * args.patch_width * args.out_channels[-1], args.dim)
        )
        # Transformer Encoder
        self.n_blocks = nn.ModuleList()
        for i in range(args.n_layers):
            self.n_blocks.append(Encoder_layer(int(args.dim / args.n_heads), args.dropout, args.n_heads))
        # recover
        self.recover = Rearrange(pattern='b (n_h n_w) dim -> b dim n_h n_w',
                                 n_h=emb_height//args.patch_height,
                                 n_w=emb_width//args.patch_width)
        self.recover_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, stride=(1, 1), kernel_size=(3, 3), padding='same'))

    def forward(self, x):
        # Input shape:  [batch_size, out_channels, height, width]
        x = self.patch_emb(x)
        for block in self.n_blocks:
            x = block(x)
            # Output shape: [batch_size, n_patches, n_features]
        x = self.recover_conv2d(self.recover(x))
        # shape: [batch, channels height width]
        return x

#ViT model from ViT Notebook

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from tqdm import tqdm

def patchify(images, n_patches):
    n, c, h, w = images.shape
    assert h == w, "Patchify method is implemented for square images only"
    patch_size = h // n_patches
    patches = torch.zeros(n, n_patches**2, c * patch_size * patch_size)

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.zeros(sequence_length, d)
    for i in range(sequence_length):
        for j in range(0, d, 2):
            result[i][j] = np.sin(i / (10000 ** (j / d)))
            if j + 1 < d:
                result[i][j + 1] = np.cos(i / (10000 ** (j / d)))
    return result

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=1):
        super(MyMSA, self).__init__()
        assert d % n_heads == 0, "Embedding dim must be divisible by number of heads"
        self.d_head = d // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3*D]
        qkv = qkv.view(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, d_head]

        attn_weights = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, heads, N, N]
        attn = self.softmax(attn_weights)
        out = attn @ v  # [B, heads, N, d_head]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        return self.out_proj(out)

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MyViT(nn.Module):
    def __init__(self, input_shape, patch_size, n_blocks, hidden_d, n_heads, out_d):
        super(MyViT, self).__init__()
        c, h, w = input_shape
        assert h % patch_size == 0 and w % patch_size == 0, "Image must be divisible by patch size"

        self.patch_size = patch_size
        self.n_patches_h = h // patch_size
        self.n_patches_w = w // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        patch_dim = c * patch_size * patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, hidden_d)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, hidden_d))

        # Transformer encoder (custom)
        self.blocks = nn.Sequential(
            *[MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_d),
            nn.Linear(hidden_d, out_d)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size, self.patch_size

        # Split into patches
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)  # [B, C, n_ph, n_pw, ph, pw]
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, n_ph, n_pw, C, ph, pw]
        patches = patches.contiguous().view(B, self.n_patches, -1)  # [B, n_patches, patch_dim]

        # Embed + position
        x = self.patch_embedding(patches) + self.pos_embedding  # [B, n_patches, hidden_d]

        # Custom transformer blocks
        x = self.blocks(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        return self.mlp_head(x)

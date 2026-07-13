import torch
import torch.nn as nn
import warnings
from typing import Optional

def window_partition_1d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partitions a 1D sequence of shape (B, L, C) into windows of shape (B * num_windows, window_size, C).
    
    Args:
        x: Input tensor of shape (B, L, C)
        window_size: Size of the partitioning window
        
    Returns:
        Tensor of shape (B * (L // window_size), window_size, C)
    """
    B, L, C = x.shape
    assert L % window_size == 0, f"Sequence length {L} must be divisible by window size {window_size}"
    num_windows = L // window_size
    # Reshape to (B, num_windows, window_size, C)
    x = x.view(B, num_windows, window_size, C)
    # Reshape to (B * num_windows, window_size, C)
    windows = x.view(-1, window_size, C)
    return windows

def window_reverse_1d(windows: torch.Tensor, window_size: int, seq_len: int) -> torch.Tensor:
    """
    Reconstructs the original sequence of shape (B, seq_len, C) from windows.
    
    Args:
        windows: Partitioned windows of shape (B * num_windows, window_size, C)
        window_size: Size of the partitioning window
        seq_len: Original sequence length
        
    Returns:
        Tensor of shape (B, seq_len, C)
    """
    num_windows = seq_len // window_size
    B_num_windows = windows.shape[0]
    B = B_num_windows // num_windows
    C = windows.shape[2]
    # Reshape to (B, num_windows, window_size, C)
    x = windows.view(B, num_windows, window_size, C)
    # Reshape to (B, seq_len, C)
    x = x.view(B, seq_len, C)
    return x

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in representation of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in representation of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention1D(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (num_windows * B, window_size, C)
            mask: Optional attention mask of shape (num_windows, window_size, window_size)
            position_bias: Optional position bias of shape (num_heads, window_size, window_size)
        """
        B_part, N, C = x.shape
        # qkv shape: (3, B_part, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B_part, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if position_bias is not None:
            attn = attn + position_bias

        if mask is not None:
            nW = mask.shape[0]
            B = B_part // nW
            # repeat mask along batch dimension and add head dimension
            mask = mask.repeat(B, 1, 1).unsqueeze(1)  # (B * nW, 1, N, N)
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_part, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchMerging1D(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(2 * dim)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (B, L, C)
        """
        B, L, C = x.shape
        if L % 2 != 0:
            warnings.warn(f"Sequence length L={L} is odd, truncating to L={L-1}")
            x = x[:, :L-1, :]
            L = L - 1

        x0 = x[:, 0::2, :]  # (B, L//2, C)
        x1 = x[:, 1::2, :]  # (B, L//2, C)
        x = torch.cat([x0, x1], dim=-1)  # (B, L//2, 2*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinBlock1D(nn.Module):
    def __init__(self, dim: int, input_resolution: Optional[int], num_heads: int,
                 window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if self.shift_size < 0 or self.shift_size >= self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention1D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (B, L, C)
        """
        B, L, C = x.shape
        
        # Calculate padding needed to make L divisible by window_size
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        if pad_l > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_l))
            
        B, L_pad, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Determine shift size dynamically based on sequence length
        if self.shift_size > 0:
            if L_pad <= self.window_size:
                shift_size = 0
            else:
                shift_size = self.shift_size
        else:
            shift_size = 0

        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=-shift_size, dims=1)
            
            # Construct attention mask dynamically if shape/device changes
            if (not hasattr(self, 'attn_mask') or 
                self.attn_mask is None or 
                self.attn_mask.shape[0] != (L_pad // self.window_size) or 
                self.attn_mask.device != x.device):
                
                img_mask = torch.zeros((1, L_pad, 1), device=x.device)
                slices = (slice(0, -self.window_size),
                          slice(-self.window_size, -shift_size),
                          slice(-shift_size, None))
                cnt = 0
                for s in slices:
                    img_mask[:, s, :] = cnt
                    cnt += 1
                
                mask_windows = window_partition_1d(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
                
                if hasattr(self, '_buffers') and 'attn_mask' in self._buffers:
                    self._buffers['attn_mask'] = attn_mask
                else:
                    self.register_buffer("attn_mask", attn_mask, persistent=False)
            
            mask = self.attn_mask
        else:
            shifted_x = x
            mask = None

        # Partition windows
        x_windows = window_partition_1d(shifted_x, self.window_size)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=mask)

        # Reverse window partition
        shifted_x = window_reverse_1d(attn_windows, self.window_size, L_pad)

        # Reverse shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=shift_size, dims=1)
        else:
            x = shifted_x

        # Residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Slice back to original length if padded
        if pad_l > 0:
            x = x[:, :L, :]
        return x

class SwinStage1D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 downsample: bool = True, mlp_ratio: float = 4., qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., dpr_list: list = [], norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList()
        for j in range(depth):
            shift_size = 0 if (j % 2 == 0) else (window_size // 2)
            block = SwinBlock1D(
                dim=dim,
                input_resolution=None,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_list[j],
                norm_layer=norm_layer
            )
            self.blocks.append(block)
        
        if downsample:
            self.downsample = PatchMerging1D(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x)
        return x

class PlainSwin1D(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, depths: list, num_heads: list,
                 window_size: int = 8, mlp_ratio: float = 4., qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm, num_classes: int = 30):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Patch projection/embedding
        self.patch_embed = nn.Linear(input_dim, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers/stages
        self.stages = nn.ModuleList()
        curr_dim = embed_dim
        for i_layer in range(self.num_layers):
            is_last = (i_layer == self.num_layers - 1)
            stage_dpr = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            
            stage = SwinStage1D(
                dim=curr_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                downsample=not is_last,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dpr_list=stage_dpr,
                norm_layer=norm_layer
            )
            self.stages.append(stage)
            if not is_last:
                curr_dim = curr_dim * 2
                
        self.norm = norm_layer(curr_dim)
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (B, L, input_dim)
        """
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for stage in self.stages:
            x = stage(x)
            
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

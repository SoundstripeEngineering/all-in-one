"""
NATTEN compatibility layer for version 0.21+.

Implements neighborhood attention using pure PyTorch to provide backward
compatibility with the old natten API (natten1dqkrpb, natten1dav, etc.)

This module is used to patch allin1's dinat.py at runtime since newer
versions of natten removed the legacy API functions.
"""

import torch
import torch.nn.functional as F


def na1d_qk(query, key, kernel_size, dilation, rpb=None):
    """
    Compute 1D neighborhood attention scores (QK^T + RPB).

    Args:
        query: [batch, heads, seqlen, head_dim] - already permuted and scaled by caller
        key: [batch, heads, seqlen, head_dim]
        kernel_size: Size of attention window
        dilation: Spacing between attended positions
        rpb: Relative positional bias [heads, 2*kernel_size-1]

    Returns:
        attention_scores: [batch, heads, seqlen, kernel_size]
    """
    batch, heads, seqlen, head_dim = query.shape

    half_k = kernel_size // 2
    pad_size = half_k * dilation

    # Pad key sequence: [batch, heads, seqlen+2*pad_size, head_dim]
    key_padded = F.pad(key, (0, 0, pad_size, pad_size), mode='replicate')

    # Gather neighbor keys
    if dilation == 1:
        # Use unfold for efficiency when dilation=1
        # unfold on dim 2 (seqlen): [batch, heads, seqlen, head_dim, kernel_size]
        neighbor_keys = key_padded.unfold(2, kernel_size, 1)
        # Permute to: [batch, heads, seqlen, kernel_size, head_dim]
        neighbor_keys = neighbor_keys.permute(0, 1, 2, 4, 3)
    else:
        # For dilation > 1, manually gather
        neighbor_list = []
        for k_idx in range(kernel_size):
            offset = (k_idx - half_k) * dilation + pad_size
            neighbor_list.append(key_padded[:, :, offset:offset+seqlen, :])
        neighbor_keys = torch.stack(neighbor_list, dim=3)  # [batch, heads, seqlen, kernel_size, head_dim]

    # Compute attention scores: query @ neighbor_keys^T
    # query: [batch, heads, seqlen, head_dim]
    # neighbor_keys: [batch, heads, seqlen, kernel_size, head_dim]
    # Result: [batch, heads, seqlen, kernel_size]
    attn_scores = torch.einsum('bhsd,bhskd->bhsk', query, neighbor_keys)

    # Add relative positional bias if provided
    if rpb is not None:
        # rpb: [heads, 2*kernel_size-1]
        # For neighborhood attention, we use the center kernel_size entries
        center = kernel_size - 1
        start_idx = center - half_k
        end_idx = center + half_k + 1

        if rpb.shape[1] >= end_idx and dilation == 1:
            rpb_slice = rpb[:, start_idx:end_idx]  # [heads, kernel_size]
            # Expand to [1, heads, 1, kernel_size] for broadcasting
            attn_scores = attn_scores + rpb_slice.unsqueeze(0).unsqueeze(2)

    return attn_scores


def na1d_av(attn_probs, value, kernel_size, dilation):
    """
    Apply 1D neighborhood attention weights to values.

    Args:
        attn_probs: [batch, heads, seqlen, kernel_size] - attention probabilities
        value: [batch, heads, seqlen, head_dim]
        kernel_size: Size of attention window
        dilation: Spacing between attended positions

    Returns:
        output: [batch, heads, seqlen, head_dim]
    """
    batch, heads, seqlen, head_dim = value.shape

    half_k = kernel_size // 2
    pad_size = half_k * dilation

    # Pad values: [batch, heads, seqlen+2*pad_size, head_dim]
    value_padded = F.pad(value, (0, 0, pad_size, pad_size), mode='replicate')

    # Gather neighbor values
    if dilation == 1:
        # unfold on dim 2: [batch, heads, seqlen, head_dim, kernel_size]
        neighbor_values = value_padded.unfold(2, kernel_size, 1)
        # Permute to: [batch, heads, seqlen, kernel_size, head_dim]
        neighbor_values = neighbor_values.permute(0, 1, 2, 4, 3)
    else:
        neighbor_list = []
        for k_idx in range(kernel_size):
            offset = (k_idx - half_k) * dilation + pad_size
            neighbor_list.append(value_padded[:, :, offset:offset+seqlen, :])
        neighbor_values = torch.stack(neighbor_list, dim=3)  # [batch, heads, seqlen, kernel_size, head_dim]

    # Weighted sum: [batch, heads, seqlen, head_dim]
    # attn_probs: [batch, heads, seqlen, kernel_size]
    # neighbor_values: [batch, heads, seqlen, kernel_size, head_dim]
    output = torch.einsum('bhsk,bhskd->bhsd', attn_probs, neighbor_values)

    return output


def na2d_qk(query, key, kernel_size, dilation, rpb=None):
    """
    Compute 2D neighborhood attention scores.

    Args:
        query: [batch, heads, height, width, head_dim] - already permuted and scaled by caller
        key: [batch, heads, height, width, head_dim]
        kernel_size: Size of attention window (int or tuple)
        dilation: Spacing (int or tuple)
        rpb: Relative positional bias [heads, 2*kernel_size-1, 2*kernel_size-1]

    Returns:
        attention_scores: [batch, heads, height, width, kernel_size*kernel_size]
    """
    batch, heads, height, width, head_dim = query.shape

    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    if isinstance(dilation, int):
        dh = dw = dilation
    else:
        dh, dw = dilation

    half_kh, half_kw = kh // 2, kw // 2
    pad_h, pad_w = half_kh * dh, half_kw * dw

    # Pad key: [batch, heads, height+2*pad_h, width+2*pad_w, head_dim]
    key_padded = F.pad(key, (0, 0, pad_w, pad_w, pad_h, pad_h), mode='replicate')

    # Gather neighbors
    neighbor_list = []
    for i in range(kh):
        for j in range(kw):
            h_offset = (i - half_kh) * dh + pad_h
            w_offset = (j - half_kw) * dw + pad_w
            neighbor_list.append(key_padded[:, :, h_offset:h_offset+height, w_offset:w_offset+width, :])

    neighbor_keys = torch.stack(neighbor_list, dim=4)  # [batch, heads, height, width, kh*kw, head_dim]

    # Compute attention scores
    # query: [batch, heads, height, width, head_dim]
    # neighbor_keys: [batch, heads, height, width, kh*kw, head_dim]
    # Result: [batch, heads, height, width, kh*kw]
    attn_scores = torch.einsum('bhwxd,bhwxkd->bhwxk', query, neighbor_keys)

    # Add RPB if provided
    if rpb is not None and dh == 1 and dw == 1:
        ch, cw = kh - 1, kw - 1
        if rpb.shape[1] >= ch + half_kh + 1 and rpb.shape[2] >= cw + half_kw + 1:
            rpb_slice = rpb[:, ch-half_kh:ch+half_kh+1, cw-half_kw:cw+half_kw+1]
            rpb_flat = rpb_slice.reshape(heads, kh * kw)
            # Expand to [1, heads, 1, 1, kh*kw] for broadcasting
            attn_scores = attn_scores + rpb_flat.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    return attn_scores


def na2d_av(attn_probs, value, kernel_size, dilation):
    """
    Apply 2D neighborhood attention weights to values.

    Args:
        attn_probs: [batch, heads, height, width, kernel_size*kernel_size]
        value: [batch, heads, height, width, head_dim]
        kernel_size: Size of attention window
        dilation: Spacing

    Returns:
        output: [batch, heads, height, width, head_dim]
    """
    batch, heads, height, width, head_dim = value.shape

    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    if isinstance(dilation, int):
        dh = dw = dilation
    else:
        dh, dw = dilation

    half_kh, half_kw = kh // 2, kw // 2
    pad_h, pad_w = half_kh * dh, half_kw * dw

    # Pad value: [batch, heads, height+2*pad_h, width+2*pad_w, head_dim]
    value_padded = F.pad(value, (0, 0, pad_w, pad_w, pad_h, pad_h), mode='replicate')

    # Gather neighbors
    neighbor_list = []
    for i in range(kh):
        for j in range(kw):
            h_offset = (i - half_kh) * dh + pad_h
            w_offset = (j - half_kw) * dw + pad_w
            neighbor_list.append(value_padded[:, :, h_offset:h_offset+height, w_offset:w_offset+width, :])

    neighbor_values = torch.stack(neighbor_list, dim=4)  # [batch, heads, height, width, kh*kw, head_dim]

    # Weighted sum
    # attn_probs: [batch, heads, height, width, kh*kw]
    # neighbor_values: [batch, heads, height, width, kh*kw, head_dim]
    output = torch.einsum('bhwxk,bhwxkd->bhwxd', attn_probs, neighbor_values)

    return output


def patch_allin1_natten():
    """
    Patch allin1's dinat module to use our compatibility functions.

    Call this before using allin1.analyze() to ensure compatibility
    with natten 0.21+.
    """
    try:
        from allin1.models import dinat

        # Patch the module-level imports that dinat expects
        dinat.na1d_qk = na1d_qk
        dinat.na1d_av = na1d_av
        dinat.na2d_qk = na2d_qk
        dinat.na2d_av = na2d_av

        return True
    except ImportError:
        return False

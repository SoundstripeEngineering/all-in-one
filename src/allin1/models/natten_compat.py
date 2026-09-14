"""Legacy non-causal neighborhood attention implemented with PyTorch.

Match NATTEN's shifted boundary windows and relative positional bias, including
for dilated neighborhoods. Callers pad axes shorter than kernel * dilation.
"""
import torch


def _neighbors(length, kernel, dilation, device):
    if kernel < 1 or kernel % 2 == 0 or dilation < 1:
        raise ValueError('kernel must be positive and odd; dilation must be positive')
    if length < kernel * dilation:
        raise ValueError('attention axis must be at least kernel * dilation')
    positions = torch.arange(length, device=device)
    residue = positions % dilation
    lane_length = (length - 1 - residue) // dilation + 1
    start = torch.minimum((positions // dilation - kernel // 2).clamp(min=0), lane_length - kernel)
    indices = residue[:, None] + (start[:, None] + torch.arange(kernel, device=device)) * dilation
    bias_indices = kernel - 1 + (indices - positions[:, None]) // dilation
    return indices, bias_indices


def na1d_qk(query, key, kernel_size, dilation, rpb=None):
    indices, bias = _neighbors(query.shape[2], kernel_size, dilation, query.device)
    scores = torch.einsum('bhsd,bhskd->bhsk', query, key[:, :, indices, :])
    if rpb is not None:
        scores = scores + rpb[:, bias].unsqueeze(0)
    return scores


def na1d_av(attn_probs, value, kernel_size, dilation):
    indices, _ = _neighbors(value.shape[2], kernel_size, dilation, value.device)
    return torch.einsum('bhsk,bhskd->bhsd', attn_probs, value[:, :, indices, :])


def _pair(value):
    return (value, value) if isinstance(value, int) else value


def _neighbors_2d(tensor, kernel_size, dilation):
    height, width = tensor.shape[2:4]
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    rows, row_bias = _neighbors(height, kh, dh, tensor.device)
    cols, col_bias = _neighbors(width, kw, dw, tensor.device)
    indices = (rows[:, None, :, None] * width + cols[None, :, None, :]).reshape(height, width, kh * kw)
    return indices, row_bias, col_bias


def na2d_qk(query, key, kernel_size, dilation, rpb=None):
    indices, row_bias, col_bias = _neighbors_2d(query, kernel_size, dilation)
    neighbors = key.flatten(2, 3)[:, :, indices, :]
    scores = torch.einsum('bhyxd,bhyxkd->bhyxk', query, neighbors)
    if rpb is not None:
        bias = rpb[:, row_bias[:, None, :, None], col_bias[None, :, None, :]]
        scores = scores + bias.flatten(-2).unsqueeze(0)
    return scores


def na2d_av(attn_probs, value, kernel_size, dilation):
    indices, _, _ = _neighbors_2d(value, kernel_size, dilation)
    neighbors = value.flatten(2, 3)[:, :, indices, :]
    return torch.einsum('bhyxk,bhyxkd->bhyxd', attn_probs, neighbors)


def patch_allin1_natten():
    try:
        from allin1.models import dinat
        dinat.na1d_qk = na1d_qk
        dinat.na1d_av = na1d_av
        dinat.na2d_qk = na2d_qk
        dinat.na2d_av = na2d_av
        return True
    except ImportError:
        return False

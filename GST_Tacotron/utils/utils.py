import torch
import numpy as np

def make_pad_mask(lengths, maxlen=None):
    """Make mask for padding frames
    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.
    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask

def pad_1d(x, max_len, constant_values=0):
    """Pad a 1d-tensor.
    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0
    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        (0, max_len - len(x)),
        mode="constant",
        constant_values=constant_values,
    )
    return x

def pad_2d(x, max_len, constant_values=0):
    """Pad a 2d-tensor.
    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0
    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x
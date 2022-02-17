import torch

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
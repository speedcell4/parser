# -*- coding: utf-8 -*-

import os
import sys
import unicodedata
import urllib
import zipfile

import torch


def ispunct(token):
    return all(unicodedata.category(char).startswith('P') for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A'] for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char) for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char) for char in token)


def tohalfwidth(token):
    return unicodedata.normalize('NFKC', token)


def stripe(x, n, w, offset=(0, 0), horizontal=True, dim1=-2, dim2=-1):
    r"""
    Returns a diagonal stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the two dims.
        horizontal (bool): ``True`` if returns a horizontal stripe; ``False`` otherwise. Default: ``True``.
        dim1 (int) – first dim with respect to which to take stripe. Default: -2.
        dim2 (int) – second dim with respect to which to take stripe. Default: -1.

    Returns:
        a diagonal stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """

    if dim1 > dim2:
        x = x.transpose(dim1, dim2)
    x = x.contiguous()
    dim1, dim2 = sorted(i if i >= 0 else x.dim() + i for i in (dim1, dim2))
    size = list(x.shape)
    size[dim1], size[dim2] = n, w
    slices = [slice(None)] * x.dim()
    slices[dim1], slices[dim2] = offset[0], offset[1]
    first = x[tuple(slices)]
    length = x.size(dim2)
    stride, numel = list(x.stride()), x[tuple([*[0]*dim2, offset[1]])].numel()
    stride[dim1], stride[dim2] = (length + 1) * numel, (1 if horizontal else length) * numel
    return x.as_strided(size=size, stride=stride, storage_offset=first.storage_offset())


def pad(tensors, padding_value=0, total_length=None, padding_side='right'):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def download(url, reload=False):
    path = os.path.join(os.path.expanduser('~/.cache/supar'), os.path.basename(urllib.parse.urlparse(url).path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if reload:
        os.remove(path) if os.path.exists(path) else None
    if not os.path.exists(path):
        sys.stderr.write(f"Downloading: {url} to {path}\n")
        try:
            torch.hub.download_url_to_file(url, path, progress=True)
        except urllib.error.URLError:
            raise RuntimeError(f"File {url} unavailable. Please try other sources.")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            members = f.infolist()
            path = os.path.join(os.path.dirname(path), members[0].filename)
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile.')
            if reload or not os.path.exists(path):
                f.extractall(os.path.dirname(path))
    return path

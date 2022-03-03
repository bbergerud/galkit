import math
import numpy
import torch
from typing import Optional, Union

def group_by(groups, values, reducer:Optional[callable] = None):
    is_tensor = isinstance(groups, torch.Tensor)
    to_type = torch.as_tensor if is_tensor else numpy.asarray
    unique = (torch if is_tensor else numpy).unique(groups)

    collection = {}
    for u in unique:
        mask = groups == u
        subvalues = values[mask]
        if reducer is not None:
            subvalues = reducer(values)
        else:
            subvalues = (torch if is_tensor else numpy).nanmean(subvalues)
        collection[u] = subvalues

    return unique, to_type([v for v in collection.values()])

def bin2map(
    binid_map : Union[numpy.ndarray, torch.Tensor],
    value_bin : Union[numpy.ndarray, torch.Tensor],
    binid_start : int = 0,
):
    is_tensor = isinstance(value_bin, torch.Tensor)
    full_like = torch.full_like if is_tensor else numpy.full_like

    bin1d = binid_map.flatten()
    val1d = full_like(bin1d, numpy.nan, dtype=torch.float32 if is_tensor else 'float')
    mask  = bin1d >= binid_start

    val1d[mask] = value_bin[bin1d[mask] - binid_start]
    return val1d.reshape(binid_map.shape)

def map2bin(
    binid_map : Union[numpy.ndarray, torch.Tensor],
    value_map : Union[numpy.ndarray, torch.Tensor],
    weight_map : Optional[Union[numpy.ndarray, torch.Tensor]] = None,
    binid_start : int = 0,
    reducer:Optional[callable] = None
) -> Union[numpy.ndarray, torch.Tensor]:

    bin1d = binid_map.flatten()
    val1d = value_map.flatten()

    binid_bin, value_bin = group_by(bin1d, val1d, reducer)

    value_bin = numpy.asarray(value_bin)
    binid_bin = numpy.asarray(binid_bin, dtype=int)
    mask = binid_bin >= binid_start

    return value_bin[mask], binid_bin[mask]
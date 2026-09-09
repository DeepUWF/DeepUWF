"""Partial, sample-count weighted state-dictionary aggregation."""

from collections import OrderedDict
import random

import torch


def select_shareable_keys(state_dict, fraction, seed):
    """Select a reproducible subset of floating-point tensors for one round."""
    if not 0 < fraction <= 1:
        raise ValueError("share fraction must be in (0, 1]")
    keys = [name for name, value in state_dict.items() if torch.is_floating_point(value)]
    count = max(1, round(len(keys) * fraction))
    return random.Random(seed).sample(keys, count)


def aggregate_partial_state(local_state, updates):
    """Return local state with each received tensor averaged by sample count.

    ``updates`` is an iterable of dictionaries with ``sample_count`` and a
    ``parameters`` mapping.  Keys not received remain local, which implements
    partial parameter sharing without transmitting model buffers or labels.
    """
    result = OrderedDict((key, value.detach().clone()) for key, value in local_state.items())
    grouped = {}
    for update in updates:
        weight = int(update["sample_count"])
        if weight <= 0:
            continue
        for key, value in update["parameters"].items():
            if key not in result or not torch.is_floating_point(result[key]):
                continue
            grouped.setdefault(key, []).append((value, weight))
    for key, tensors in grouped.items():
        total = sum(weight for _, weight in tensors)
        if total:
            value = sum(tensor.to(dtype=result[key].dtype) * weight for tensor, weight in tensors) / total
            result[key] = value.to(device=result[key].device)
    return result

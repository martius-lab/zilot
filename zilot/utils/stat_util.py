import functools
import warnings
from typing import NamedTuple

import numpy as np


def bootstrap_confidence_interval(data: np.ndarray, stat_fn, quantile=0.95, n_samples=1000) -> tuple:
    n = len(data)
    samples = np.array([stat_fn(data[np.random.choice(n, n)]) for _ in range(n_samples)])
    mean = np.mean(samples)
    lower = np.quantile(samples, 1 - quantile)
    upper = np.quantile(samples, quantile)
    return dict(mean=mean, lower=lower, upper=upper)


AGG_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    "ci.95": lambda x: bootstrap_confidence_interval(x, np.mean, 0.95),
    "count": len,
    "first": lambda x: x[0],
    "last": lambda x: x[-1],
    "max": max,
    "min": min,
    "sum": sum,
    "any": any,
    "all": all,
    "cat": lambda x: np.concatenate(list(x)),
    "stack": lambda x: x if isinstance(x, np.ndarray) else np.stack(x),
}


class Agg(NamedTuple):
    fn: str
    keys: str | tuple[str]


def agg(data: dict[str, np.ndarray], agg: Agg) -> np.ndarray:
    fn, keys = agg
    if isinstance(fn, str):
        fn = AGG_FUNCS[fn]
    if isinstance(keys, str):
        x = data[keys]
        if isinstance(x, dict):
            return fn(**x)
        return fn(x)
    else:
        return fn(**{k: data[k] for k in keys})


_WARNED_USER_FOR_KEY = set()


def aggregate(data: dict[str, np.ndarray], **aggs):
    res = {}
    for k, v in aggs.items():
        try:
            res[k] = agg(data, v)
        except Exception as e:
            if k not in _WARNED_USER_FOR_KEY:
                warnings.warn(f"Failed to aggregate {k}: {e.__class__.__name__}:{e}")
                _WARNED_USER_FOR_KEY.add(k)
    return res


def stack_nested_dict(data: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = functools.reduce(lambda a, b: a | b, (set(d.keys()) for d in data.values()))
    order = sorted(keys)
    out = {}
    for k in order:
        x = [d[k] for d in data.values() if k in d.keys()]
        try:
            out[k] = np.stack(x)
        except Exception:  # probably shape missmatch
            out[k] = x
    return out

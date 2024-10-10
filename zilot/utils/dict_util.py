from typing import Any, Callable, Dict

from tensordict import TensorDict, TensorDictBase


def _apply_td(f: Callable, d: TensorDictBase) -> TensorDictBase:
    return d.apply(f)


def _apply_dict(f: Callable, d: Dict[str, Any]) -> Dict[str, Any]:
    d_out = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            d_out[k] = _apply_dict(f, v)
        else:
            d_out[k] = f(v)
    return d_out


def _apply_td_(f: Callable, d: TensorDictBase) -> TensorDictBase:
    return d.apply_(f)


def _apply_dict_(f: Callable, d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _apply_dict_(f, v)
        else:
            try:
                d[k] = f(v)
            except Exception as e:
                raise ValueError(f"Failed to apply function to {k}: {e}")
    return d


def apply(f: Callable, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a function to all leafs.

    Args:
        f: function to apply
        d: dictionary
    """
    if isinstance(d, TensorDictBase):
        return _apply_td(f, d)
    else:
        return _apply_dict(f, d)


def apply_(f: Callable, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a function to all leafs

    Args:
        f: function to apply
        d: dictionary
    """
    if isinstance(d, TensorDictBase):
        return _apply_td_(f, d)
    else:
        return _apply_dict_(f, d)


def map(f: Dict[str, Callable], d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a dictionary of functions to a dictionary of tensors.
    Applies f[k] to d[k] if k in f.

    Args:
        f: dictionary of functions
        d: dictionary
    """
    d_out = dict()
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, TensorDictBase):
            d_out[k] = map(f, v)
        elif k in f:
            d_out[k] = f[k](v)
        else:
            d_out[k] = v
    if isinstance(d, TensorDictBase):
        try:
            return TensorDict(d_out, device=d.device)
        except Exception:
            return TensorDict(d_out, device=d.device, batch_size=d.batch_size)
    else:
        return d_out


def _map_td_(f: Dict[str, Callable], d: TensorDict) -> TensorDict:
    """
    Map a dictionary of functions to a dictionary of tensors in place.
    Applies f[k] to d[k] if k in f.

    Args:
        f: dictionary of functions
        d: dictionary
    """
    for k, v in d.items():
        if isinstance(v, TensorDictBase):
            d.set_(k, _map_td_(f, v))
        elif k in f:
            d.set_(k, f[k](v))
    return d


def _map_dict_(f: Dict[str, Callable], d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a dictionary of functions to a dictionary of tensors in place.
    Applies f[k] to d[k] if k in f.

    Args:
        f: dictionary of functions
        d: dictionary
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _map_dict_(f, v)
        elif k in f:
            d[k] = f[k](v)
    return d


def map_(f: Dict[str, Callable], d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a dictionary of functions to a dictionary of tensors in place.
    Applies f[k] to d[k] if k in f.

    Args:
        f: dictionary of functions
        d: dictionary
    """
    if isinstance(d, TensorDictBase):
        return _map_td_(f, d)
    else:
        return _map_dict_(f, d)


def fold(d: Dict[str, Any], f_node: Callable[[Dict], Any], f_leaf: Callable[[Any], Any] = lambda x: x) -> Any:
    """
    Fold a dictionary of tensors into a single value.

    Args:
        d: dictionary
        f_leaf: function to apply to leaf tensors
        f_node: function to apply to non-leaf nodes (default: identity function)
    """
    if isinstance(d, dict) or isinstance(d, TensorDictBase):
        d_out = dict()
        for k, v in d.items():
            d_out[k] = fold(v, f_node, f_leaf)
        return f_node(d_out)
    else:
        return f_leaf(d)


def flatten(d: Dict[str, Any], sep="/") -> Dict[str, Any]:
    """
    Flatten a dictionary of dictionaries into a single dictionary.

    Args:
        d: dictionary
    """

    def f_node(d):
        d_out = dict()
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    d_out[f"{k}{sep}{k2}"] = v2
            else:
                d_out[k] = v
        return d

    return fold(d, f_node)

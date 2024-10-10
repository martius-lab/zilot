import contextlib

import numpy as np
import zarr
import zarr.storage


def _parse_memspec(s: str) -> int:
    """Parse a memory specification string into bytes."""
    s = s.upper()
    if s[-2:] == "KB":
        return int(s[:-2]) * 1024
    if s[-2:] == "MB":
        return int(s[:-2]) * 1024**2
    if s[-2:] == "GB":
        return int(s[:-2]) * 1024**3
    if s[-2:] == "TB":
        return int(s[:-2]) * 1024**4
    if s[-1] == "B":
        return int(s[:-1])
    return int(s)


class IdCollector:
    def __init__(self, storage: zarr.storage.Store, max_ram="1GB", **dset_kwargs):
        self._store = storage
        self._root = zarr.group(store=self._store, overwrite=True)
        self._max_ram = _parse_memspec(max_ram)
        self._registered_dsets = dict()
        self._dset_kwargs = dset_kwargs

        # variable state
        self._ram = 0
        self._cache = []
        self._attr_cache = []

    def add(self, key: str, id: np.ndarray, data: np.ndarray):
        if data.dtype == object:
            raise ValueError(f"`data.dtype` `object` currently not supported (key={key})")

        self._ram += id.nbytes + data.nbytes
        self._cache.append((key, id, data))

        if self._ram > self._max_ram // 2:
            self.flush()

    def add_attr(self, key: str, attr_name: str, value):
        self._attr_cache.append((key, attr_name, value))

    def register_key(self, key: str, strict: bool = True, **dset_kwargs):
        if key in self._registered_dsets:
            raise ValueError(f"Key {key} already registered")
        dset_kwargs["strict"] = strict
        self._registered_dsets[key] = dset_kwargs

    def flush(self):
        self._ram = 0

        x: dict[int, dict[str, list[np.ndarray]]] = {}
        for key, id, data in self._cache:
            for i, d in zip(id, data):
                x.setdefault(int(i), {}).setdefault(key, []).append(d)
        self._cache = []

        # write data
        for i, dset in x.items():
            for key, data in dset.items():
                strict = True
                dset_kwargs = self._dset_kwargs.copy()
                if key in self._registered_dsets:
                    strict = self._registered_dsets[key]["strict"]
                    dset_kwargs.update(self._registered_dsets[key])
                    del dset_kwargs["strict"]
                if strict:
                    try:
                        data = np.stack(data)
                    except ValueError as e:
                        raise ValueError(
                            f"Cannot stack data for `{i}/{key}`. Shapes occuring are {set([x.shape for x in data])}."
                        ) from e
                    key = str(i) + "/" + key
                    if key in self._root:
                        self._root[key].append(data)
                    else:
                        self._root.create_dataset(key, data=data, **dset_kwargs)
                else:
                    key = str(i) + "/" + key
                    idx = 0 if key + "/0" not in self._root else len(self._root[key].keys())
                    for j, d in enumerate(data):
                        self._root.create_dataset(key + "/" + str(idx + j), data=d, **dset_kwargs)
                    self._root[key].attrs["_leaf"] = True
        del x

        # write attributes
        for key, attr_name, value in self._attr_cache:
            self._root[key].attrs[attr_name] = value
        self._attr_cache = []

    def close(self):
        self.flush()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EpCollector:
    def __init__(self, *args, **kwargs):
        self._collector = IdCollector(*args, **kwargs)
        self.flush = self._collector.flush
        self.close = self._collector.close
        self.register_key = self._collector.register_key
        self._cur_ep = None

    def reset(self):
        self._cur_ep = self._cur_ep + 1 if self._cur_ep is not None else 0

    def add(self, data: dict[str, np.ndarray]):
        for k, v in data.items():
            self._add(k, v)

    def _add(self, key: str, data: np.ndarray | dict[str, np.ndarray]):
        if isinstance(data, dict):
            for k, v in data.items():
                self._add(key + "/" + k, v)
            return
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if self._cur_ep is None:
            self.reset()  # in case user does not call `reset` first
        self._collector.add(key, np.array([self._cur_ep]), data[np.newaxis, ...])

    def add_attr(self, key: str, attr_name: str, value):
        key = str(self._cur_ep) + "/" + key
        self._collector.add_attr(key, attr_name, value)

    def __enter__(self):
        self._collector.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collector.__exit__(exc_type, exc_val, exc_tb)


class BatchedEpCollector:
    def __init__(self, batch_size: int, *args, **kwargs):
        self._collector = IdCollector(*args, **kwargs)
        self.flush = self._collector.flush
        self.close = self._collector.close
        self.register_key = self._collector.register_key
        self._batch_size = batch_size
        self._ids = None

    def reset(self, ids: np.ndarray | None = None):
        if self._ids is None:
            if ids is not None:
                raise ValueError("ids must be None on first reset")
            self._ids = np.arange(self._batch_size)
            return
        if ids is None:
            ids = np.arange(self._batch_size)
        max_id = self._ids.max()
        n = self._ids[ids].shape[0]
        self._ids[ids] = np.arange(max_id + 1, max_id + 1 + n)

    def add(self, data: dict[str, np.ndarray]):
        for k, v in data.items():
            self._add(k, v)

    def _add(self, key: str, data: np.ndarray | dict[str, np.ndarray]):
        if isinstance(data, dict):
            for k, v in data.items():
                self.add(key + "/" + k, v)
            return
        if data.shape[0] != self._batch_size:
            raise ValueError("`data.shape[0]` must be equal to `batch_size`")
        if self._ids is None:
            self.reset()  # in case user does not call `reset` first
        self._collector.add(key, self._ids.copy(), data)

    def add_attr(self, key: str, attr_name: str, value: np.ndarray):
        if value.shape[0] != self._batch_size:
            raise ValueError("`value.shape[0]` must be equal to `batch_size`")
        for i, v in zip(self._ids, value):
            self._collector.add_attr(str(i) + "/" + key, attr_name, v)

    def __enter__(self):
        self._collector.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collector.__exit__(exc_type, exc_val, exc_tb)


@contextlib.contextmanager
def collect(path, COL, STORE, **kwargs):
    store: zarr.storage.Store = STORE(path)
    try:
        with COL(store, **kwargs) as col:
            yield col
    finally:
        store.close()


def zarr_to_dict(path: str) -> dict[str, np.ndarray]:
    # TODO: write general version for any zarr/hdf5 file
    def rec(g: zarr.Group | zarr.Array):
        if isinstance(g, zarr.Array):
            return g[:]
        elif isinstance(g, zarr.Group) and g.attrs.get("_leaf", False):
            return [g[i][:] for i in range(len(g.keys()))]
        return {k: rec(v) for k, v in g.items()}

    return rec(zarr.open(path))

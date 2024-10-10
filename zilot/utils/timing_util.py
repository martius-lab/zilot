import time

import torch


class Timer:
    def __init__(self, name="", verbose=False) -> None:
        self.name = name
        self.verbose = verbose
        self.is_cuda = torch.cuda.is_available()
        self.start = torch.cuda.Event(enable_timing=True) if self.is_cuda else None
        self.end = torch.cuda.Event(enable_timing=True) if self.is_cuda else None

    def tic(self):
        if self.is_cuda:
            self.start.record()
        else:
            self.start = time.perf_counter()

    def toc(self):
        if self.is_cuda:
            self.end.record()
            torch.cuda.synchronize()
            t = self.start.elapsed_time(self.end)
        else:
            self.end = time.perf_counter()
            t = (self.end - self.start) * 1000
        if self.verbose:
            print(f"{self.name}: {t:.2f} ms")
        return t

    def __enter__(self):
        self.tic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.toc()


def get_timer(name: str) -> Timer:
    return Timer(name=name, verbose=False)


def get_running_timer(name: str) -> Timer:
    x = Timer(name=name, verbose=False)
    x.tic()
    return x

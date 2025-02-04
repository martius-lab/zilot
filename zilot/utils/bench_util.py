import torch


def benchmark(fn, inputs: list[tuple]):
    """
    Benchmark the function `fn` by running it on the inputs in `inputs` and
    returns the average time taken to run the function in miliseconds.
    """
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for input in inputs:
            _ = fn(*input)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / len(inputs)

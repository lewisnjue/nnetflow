import time
import statistics

import numpy as np
import torch
from torch import nn

from nnetflow.engine import Tensor
from nnetflow.layers import Linear


def benchmark_forward(fn, x, warmup=10, repeat=50):
    for _ in range(warmup):
        fn(x)

    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn(x)
        timings.append(time.perf_counter() - start)

    return statistics.median(timings) * 1000.0


def benchmark_case(batch_size, in_features, out_features):
    x_np = np.random.randn(batch_size, in_features).astype(np.float32)

    nnetflow_layer = Linear(in_features, out_features, bias=True, dtype=np.float32)
    nnetflow_x = Tensor(x_np.copy(), requires_grad=False, dtype=np.float32)

    torch_layer = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
    torch_x = torch.from_numpy(x_np.copy())

    nf_time = benchmark_forward(lambda _: nnetflow_layer.forward(nnetflow_x), None)
    with torch.no_grad():
        pt_time = benchmark_forward(lambda _: torch_layer(torch_x), None)

    speedup = pt_time / nf_time if nf_time > 0 else float("inf")
    return {
        "batch": batch_size,
        "in": in_features,
        "out": out_features,
        "nnetflow_ms": nf_time,
        "pytorch_ms": pt_time,
        "speedup": speedup,
    }


def main():
    sizes = [
        (64, 128, 256),
        (256, 128, 256),
        (512, 256, 512),
        (1024, 512, 1024),
        (2048, 1024, 2048),
    ]

    rows = [benchmark_case(batch, in_features, out_features) for batch, in_features, out_features in sizes]

    print("Forward-pass benchmark: nnetflow Linear vs PyTorch nn.Linear")
    print("=" * 92)
    print(f"{'batch':>8} {'in':>6} {'out':>6} {'nnetflow (ms)':>16} {'pytorch (ms)':>15} {'speedup':>10}")
    print("-" * 92)

    for row in rows:
        print(
            f"{row['batch']:>8} {row['in']:>6} {row['out']:>6} "
            f"{row['nnetflow_ms']:>15.3f} {row['pytorch_ms']:>15.3f} {row['speedup']:>9.2f}x"
        )

    print("=" * 92)
    print("Lower is better. Speedup is PyTorch time / nnetflow time.")


if __name__ == "__main__":
    main()


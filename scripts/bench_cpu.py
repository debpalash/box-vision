"""
CPU benchmark for an exported BoxVision ONNX model.

Measures three numbers and prints them as the last three lines of output.
The key names are a frozen measurement contract -- do not rename them
(the throughput key stays "throughput_ips_8w" regardless of --workers):

    latency_ms_1t:     median single-thread latency in ms
    throughput_ips_8w: aggregate images/sec of N concurrent 1-thread workers
    params_k:          parameter count (ONNX initializer elements / 1000)

Usage:
    python scripts/bench_cpu.py --onnx runs/exp1/model.onnx --input-size 320
"""

import argparse
import multiprocessing as mp
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort


def make_input(input_size: int):
    rng = np.random.default_rng(0)  # fixed seed so every run feeds identical data
    return rng.standard_normal((1, 3, input_size, input_size)).astype(np.float32)


def make_session(onnx_path: str):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])


def count_params_k(onnx_path: str) -> float:
    model = onnx.load(onnx_path)
    total = sum(int(np.prod(init.dims)) for init in model.graph.initializer)
    return total / 1000.0


def bench_latency(onnx_path: str, input_size: int, runs: int) -> float:
    """Median single-thread latency in ms over `runs` timed inferences."""
    sess = make_session(onnx_path)
    inp = make_input(input_size)
    name = sess.get_inputs()[0].name
    for _ in range(10):
        sess.run(None, {name: inp})
    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: inp})
        times_ms.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times_ms)


def throughput_worker(onnx_path, input_size, duration, barrier, queue):
    """One benchmark process: own session, warmup, then run flat-out for `duration`s."""
    sess = make_session(onnx_path)
    inp = make_input(input_size)
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: inp})
    barrier.wait()  # all workers start hammering at the same instant
    count = 0
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        sess.run(None, {name: inp})
        count += 1
    queue.put(count)


def bench_throughput(onnx_path: str, input_size: int, workers: int, duration: float) -> float:
    """Aggregate images/sec across `workers` concurrent single-thread processes."""
    ctx = mp.get_context("spawn")  # spawn, so ORT state is never forked
    barrier = ctx.Barrier(workers)
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=throughput_worker,
                    args=(onnx_path, input_size, duration, barrier, queue))
        for _ in range(workers)
    ]
    for p in procs:
        p.start()
    counts = [queue.get() for _ in range(workers)]
    for p in procs:
        p.join()
    return sum(counts) / duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    params_k = count_params_k(args.onnx)

    print(f"Benchmarking {args.onnx} @ {args.input_size}x{args.input_size}")
    print(f"  single-thread latency: median of {args.runs} runs (10 warmup)")
    latency_ms = bench_latency(args.onnx, args.input_size, args.runs)

    print(f"  throughput: {args.workers} spawned workers x {args.duration:.1f}s")
    ips = bench_throughput(args.onnx, args.input_size, args.workers, args.duration)

    # Frozen measurement contract: exact keys, last three lines, nothing after.
    print(f"latency_ms_1t:     {latency_ms:.2f}")
    print(f"throughput_ips_8w: {ips:.1f}")
    print(f"params_k:          {params_k:.1f}")


if __name__ == "__main__":
    main()

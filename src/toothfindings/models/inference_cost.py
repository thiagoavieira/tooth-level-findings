"""Parameter count, in-memory size and single-crop latency of each backbone.

Two protocols are provided:

:func:`measure`
    Batch size 1, random input at the training resolution, 10 warm-up iterations, then timed
    iterations on CPU (``iters // 5``, at least 10) and GPU (``iters``). Writes
    ``inference_cost.csv`` (by default under ``experiments/cost/``).
:func:`measure_latency_all`
    Unified single-session protocol behind the CPU/GPU latency columns of the inference-cost
    table: mean, median and 95th percentile per backbone, plus the FP16 GPU median.

Run either with an otherwise idle GPU; timings depend on the hardware.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..constants import ARCH_INPUT, TIMM_NAME
from ..utils import write_dict_csv


def model_size_mb(model) -> float:
    """Return the in-memory size of the parameters and buffers of ``model`` in MB (1e6 bytes)."""
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / 1e6


def latency_ms(model, device, size: int, iters: int = 50, warmup: int = 10) -> float:
    """Return the mean latency in milliseconds of a batch-size-1 forward pass on random input.

    Parameters
    ----------
    model : torch.nn.Module
        Model to time; moved to ``device`` and set to eval mode.
    device : torch.device
        Device to run on; CUDA is synchronised before and after the timed loop.
    size : int
        Input height and width.
    iters, warmup : int
        Timed iterations and untimed warm-up iterations.
    """
    import torch

    with torch.no_grad():
        model.eval().to(device)
        x = torch.randn(1, 3, size, size, device=device)
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def measure(out_dir: Path, iters: int = 50, archs: list[str] | None = None) -> list[dict]:
    """Measure every backbone and write ``inference_cost.csv`` into ``out_dir``.

    Parameters
    ----------
    out_dir : Path
        Output folder.
    iters : int
        Timed GPU iterations; the CPU uses ``max(10, iters // 5)``.
    archs : list of str, optional
        Backbones to measure; every key of ``ARCH_INPUT`` by default.

    Returns
    -------
    list of dict
        One row per backbone with ``arch``, ``params_M`` (millions), ``model_MB``,
        ``latency_cpu_ms``, ``latency_gpu_ms`` and ``peak_vram_MB`` (the last two are empty
        strings without a GPU).
    """
    import timm
    import torch

    rows = []
    for arch in archs or list(ARCH_INPUT):
        size = ARCH_INPUT[arch]
        model = timm.create_model(TIMM_NAME.get(arch, arch), pretrained=False, num_classes=4)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        size_mb = model_size_mb(model)
        cpu_ms = latency_ms(model, torch.device("cpu"), size, iters=max(10, iters // 5))
        gpu_ms, peak_vram = "", ""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_ms = round(latency_ms(model, torch.device("cuda"), size, iters=iters), 3)
            peak_vram = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            del model
            torch.cuda.empty_cache()
        rows.append(
            {
                "arch": arch,
                "params_M": round(params_m, 2),
                "model_MB": round(size_mb, 1),
                "latency_cpu_ms": round(cpu_ms, 2),
                "latency_gpu_ms": gpu_ms,
                "peak_vram_MB": peak_vram,
            }
        )
        print(f"  {arch:24s} params={params_m:.1f}M size={size_mb:.0f}MB cpu={cpu_ms:.1f}ms gpu={gpu_ms}ms")
    write_dict_csv(Path(out_dir) / "inference_cost.csv", rows)
    return rows


#: Tooth-level backbones of the unified latency session (label, timm id, input size).
LATENCY_MODELS = [
    ("inception_v3", "inception_v3", 299),
    ("resnetv2_50", "resnetv2_50", 224),
    ("vgg16", "vgg16", 224),
    ("convnext_tiny", "convnext_tiny.fb_in1k", 224),
    ("tf_efficientnetv2_s", "tf_efficientnetv2_s.in1k", 300),
    ("deit_small", "deit_small_patch16_224.fb_in1k", 224),
]


def bench(model, device, size: int, iters: int, warmup: int = 20, half: bool = False) -> dict:
    """Return the mean, median and 95th-percentile latency (ms) of batch-size-1 forward passes.

    Parameters
    ----------
    model : torch.nn.Module
        Model to time; converted back to FP32 afterwards when ``half`` is set.
    device : torch.device
        Device to run on; CUDA is synchronised around every timed pass.
    size : int
        Input height and width.
    iters, warmup : int
        Timed passes and untimed warm-up passes.
    half : bool
        Run in FP16.

    Returns
    -------
    dict
        ``{"mean": float, "median": float, "p95": float}`` in milliseconds.
    """
    import statistics

    import torch

    with torch.no_grad():
        model = model.eval().to(device)
        x = torch.randn(1, 3, size, size, device=device)
        if half:
            model, x = model.half(), x.half()
        times = []
        for i in range(warmup + iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000.0)
        if half:
            model.float()
    times.sort()
    return {
        "mean": statistics.fmean(times),
        "median": statistics.median(times),
        "p95": times[max(0, int(round(0.95 * len(times))) - 1)],
    }


def measure_latency_all(out_csv: Path, cpu_iters: int = 30, gpu_iters: int = 100) -> list[dict]:
    """Measure the latency of the backbones in ``LATENCY_MODELS`` and write ``out_csv``.

    These are the CPU/GPU columns of the inference-cost table. Each row has ``task``
    (always ``"task1"``, the tooth-level task), ``model``, ``input``, ``params_M``,
    ``cpu_{mean,median,p95}_ms`` and, with a GPU, ``gpu_{mean,median,p95}_ms`` and
    ``gpu_fp16_median_ms`` (``"failed:<Error>"`` when FP16 inference fails).

    Parameters
    ----------
    out_csv : Path
        Output CSV file.
    cpu_iters, gpu_iters : int
        Timed iterations per device.

    Returns
    -------
    list of dict
        The written rows.
    """
    import timm
    import torch

    rows = []
    for label, timm_id, size in LATENCY_MODELS:
        model = timm.create_model(timm_id, pretrained=False, num_classes=4)
        row = {
            "task": "task1",
            "model": label,
            "input": size,
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        }
        cpu = bench(model, torch.device("cpu"), size, cpu_iters)
        row.update({f"cpu_{k}_ms": round(v, 2) for k, v in cpu.items()})
        if torch.cuda.is_available():
            gpu = bench(model, torch.device("cuda"), size, gpu_iters)
            row.update({f"gpu_{k}_ms": round(v, 2) for k, v in gpu.items()})
            try:
                row["gpu_fp16_median_ms"] = round(
                    bench(model, torch.device("cuda"), size, gpu_iters, half=True)["median"], 2
                )
            except Exception as exc:
                row["gpu_fp16_median_ms"] = f"failed:{type(exc).__name__}"
            model.to("cpu")
            torch.cuda.empty_cache()
        rows.append(row)
        del model
    # Identifying columns first, then the latency columns alphabetically.
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("task", "model", "input", "params_M"), k))
    write_dict_csv(out_csv, rows, fields)
    return rows

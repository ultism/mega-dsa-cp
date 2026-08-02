"""Run mega-dsa-cp Phase 0.1 event-system tests on Modal (B200).

Mirrors the svdquant-kernels channel: Python sources are mounted into the
image (copy=False) and JIT-compiled on-device; no local build step.
Modal blocks counter-level profiling (ncu / nsys --gpu-metrics-device);
torch.profiler and nsys kernel timelines work — irrelevant for Phase 0.1
(latencies come from in-kernel %globaltimer).

Usage:
    modal run scripts/modal_app.py                    # smoke
    modal run scripts/modal_app.py::single_gpu        # DAG equivalence + local bench (1xB200)
    modal run scripts/modal_app.py::dual_gpu          # cross-GPU tests + bench (2xB200)

Logs: `modal run ... 2>&1 | tee log/<task>.log`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

app = modal.App("mega-dsa-cp")

# torch 2.11 cu130 + nvidia-cutlass-dsl + cuda-python: the same proven
# combo as svdquant-kernels' triton_image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("nvidia-cutlass-dsl", "cuda-python")
    .add_local_dir(
        str(ROOT / "mega_dsa_cp"),
        remote_path="/root/mega-dsa-cp/mega_dsa_cp",
        copy=False,
    )
    .add_local_dir(
        str(ROOT / "tests"),
        remote_path="/root/mega-dsa-cp/tests",
        copy=False,
    )
)


def _run(cmd: list[str], timeout: int = 300) -> None:
    """Run one command with a HARD timeout.

    Kernel execution is seconds; JIT compiles are ~40-60s each. Any command
    exceeding its ceiling is a deadlock (spin waits, NCCL, rendezvous) —
    kill it fast instead of burning GPU time. retries=0 on the functions
    keeps Modal from re-running the whole suite after a kill.
    """
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd="/root/mega-dsa-cp", timeout=timeout)


@app.function(gpu="B200", image=image, timeout=600)
def smoke() -> None:
    _run(["nvidia-smi"])
    _run(
        [
            "python",
            "-c",
            "import torch, cutlass; "
            "print('torch', torch.__version__, 'cuda', torch.version.cuda); "
            "print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); "
            "print('cutlass dsl ok')",
        ]
    )


@app.function(gpu="B200", image=image, timeout=900, retries=0)
def single_gpu() -> None:
    """Phase 0.1 single-GPU suite: random-DAG equivalence + local latency."""
    _run(["nvidia-smi"])
    for seed in ("0", "1", "2"):
        _run(["python", "tests/test_events_single_gpu.py", seed])
    _run(["python", "tests/bench_notify_latency.py", "3000"], timeout=600)


@app.function(gpu="B200:2", image=image, timeout=1200, retries=0)
def dual_gpu() -> None:
    """Phase 0.1 dual-GPU suite: ping-pong/fan-in/fan-out + cross-GPU latency.

    Two B200s on one Modal node (NVLink P2P). torchrun spawns one process
    per GPU; the symmetric buffer rendezvous goes over the LOCAL nccl
    process group.
    """
    _run(["nvidia-smi"])
    _run(
        ["torchrun", "--nproc_per_node=2", "tests/test_events_dual_gpu.py", "1000"],
        timeout=600,
    )
    _run(
        ["torchrun", "--nproc_per_node=2", "tests/bench_notify_latency.py", "3000"],
        timeout=900,
    )


@app.function(gpu="B200", image=image, timeout=900, retries=0)
def phase02_single() -> None:
    """Phase 0.2 single-GPU: static scheduler DAG equivalence, 2 phases."""
    _run(["nvidia-smi"])
    for seed in ("0", "1"):
        _run(["python", "tests/test_scheduler_single_gpu.py", seed])


@app.function(gpu="B200:2", image=image, timeout=900, retries=0)
def phase02_dual() -> None:
    """Phase 0.2 dual-GPU: cross-rank DAG through the scheduler, 2 phases."""
    _run(["nvidia-smi"])
    _run(
        ["torchrun", "--nproc_per_node=2", "tests/test_scheduler_dual_gpu.py"],
        timeout=600,
    )


@app.function(gpu="B200", image=image, timeout=900, retries=0)
def phase11_single() -> None:
    """Phase 1.1 single-GPU: FP4 paged MQA logits vs torch fp32 reference."""
    _run(["nvidia-smi"])
    _run(["python", "tests/test_logits_fp4.py", "--tiny"], timeout=420)
    _run(["python", "tests/test_logits_fp4.py"], timeout=420)
    _run(["python", "tests/test_logits_fp4.py", "--topk"], timeout=420)


@app.function(gpu="B200:2", image=image, timeout=900, retries=0)
def phase03_dual() -> None:
    """Phase 0.3 dual-GPU: bulk push + multimem primitives."""
    _run(["nvidia-smi"])
    _run(
        ["torchrun", "--nproc_per_node=2", "tests/test_comm_dual_gpu.py"],
        timeout=600,
    )


@app.function(gpu="B200:2", image=image, timeout=900, retries=0)
def phase04_dual() -> None:
    """Phase 0.4: skeleton pipeline cp=2 end-to-end (freeze point)."""
    _run(["nvidia-smi"])
    _run(
        ["torchrun", "--nproc_per_node=2", "tests/test_skeleton_cp2.py"],
        timeout=600,
    )


@app.local_entrypoint()
def main() -> None:
    smoke.remote()

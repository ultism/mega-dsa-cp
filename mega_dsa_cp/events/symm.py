"""Host-side symmetric event buffer allocation.

world == 1: plain CUDA tensor (offsets = [0]); used by cp=1 degenerate runs
and the single-GPU equivalence test.

world > 1: torch.distributed._symmetric_memory, one rendezvous per buffer.
Every rank must call this with the same num_events and group. Returns the
local tensor plus the per-peer base addresses; the device side only needs
offsets[r] = ptr[r] - ptr[my_rank] as an i64 tensor.

Compat shim for symm_mem group registration mirrors
flashinfer/comm/torch_symmetric_memory.py.
"""

import functools
from dataclasses import dataclass

import torch

from .core import CELL_BYTES


@dataclass
class EventBuffer:
    num_events: int
    rank: int
    world_size: int
    tensor: torch.Tensor  # local cells, int64 view of the u64 cells
    offsets: torch.Tensor  # i64 CUDA tensor, world_size entries
    handle: object = None  # symm_mem rendezvous handle, keeps mapping alive

    @property
    def data_ptr(self) -> int:
        return self.tensor.data_ptr()


_compat_patched = False


def _patch_group_count_reset() -> None:
    global _compat_patched
    if _compat_patched:
        return
    _compat_patched = True
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    original_destroy = dist.destroy_process_group

    @functools.wraps(original_destroy)
    def _patched_destroy(group=None):
        saved = c10d._world.group_count
        original_destroy(group)
        if group is None:
            c10d._world.group_count = saved

    dist.destroy_process_group = _patched_destroy


def _enable_symm_mem(group_name: str) -> None:
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 11):
        return
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    _patch_group_count_reset()
    enable_symm_mem_for_group(group_name)


def alloc_event_buffer(
    num_events: int,
    rank: int = 0,
    world_size: int = 1,
    group_name: str = "0",
    device: str = "cuda",
) -> EventBuffer:
    nbytes = num_events * CELL_BYTES
    numel = nbytes // 8
    if world_size == 1:
        tensor = torch.zeros(numel, dtype=torch.int64, device=device)
        offsets = torch.zeros(1, dtype=torch.int64, device=device)
        return EventBuffer(num_events, rank, world_size, tensor, offsets)

    import torch.distributed._symmetric_memory as symm_mem

    _enable_symm_mem(group_name)
    tensor = symm_mem.empty(numel, dtype=torch.int64, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    ptrs = [
        handle.get_buffer(peer, (numel,), torch.int64, storage_offset=0).data_ptr()
        for peer in range(world_size)
    ]
    base = ptrs[rank]
    offsets = torch.tensor([p - base for p in ptrs], dtype=torch.int64, device=device)
    tensor.zero_()
    return EventBuffer(num_events, rank, world_size, tensor, offsets, handle)

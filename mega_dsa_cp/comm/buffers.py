"""Host-side symmetric payload arena with named regions and phase rotation.

Layout inside one symmetric allocation:
    [phase 0: region0 | region1 | ...][phase 1: region0 | region1 | ...]
Every rank allocates the identical layout (same region sizes), so a single
per-rank offset table (base_r - base_mine) maps any local byte offset to any
peer — the same translation the event cells use.

Region sizing comes from the frozen communication inventory
(docs/overview.md 通信清单): candidates, LSE scalars, LSE partials.
"""

from dataclasses import dataclass

import torch

from ..events.symm import _enable_symm_mem


@dataclass
class SymmetricArena:
    nbytes_per_phase: int
    phases: int
    rank: int
    world_size: int
    tensor: torch.Tensor
    offsets: torch.Tensor  # i64 CUDA [world]
    mc_base: int  # NVLS multicast address, 0 when unavailable
    handle: object
    regions: dict  # name -> byte offset within one phase

    def region_off(self, name: str, phase: int = 0) -> int:
        return phase * self.nbytes_per_phase + self.regions[name]

    @property
    def data_ptr(self) -> int:
        return self.tensor.data_ptr()


def alloc_arena(
    regions: dict,
    phases: int = 2,
    rank: int = 0,
    world_size: int = 1,
    group_name: str = "0",
    device: str = "cuda",
) -> SymmetricArena:
    """regions: {name: nbytes}; each is 256B-aligned up. phases >= 2 for
    cross-step rotation."""
    offs = {}
    pos = 0
    for name, nbytes in regions.items():
        offs[name] = pos
        pos += (nbytes + 255) & ~255
    nbytes_total = pos * phases

    if world_size == 1:
        tensor = torch.zeros(nbytes_total, dtype=torch.uint8, device=device)
        offsets = torch.zeros(1, dtype=torch.int64, device=device)
        return SymmetricArena(pos, phases, rank, world_size, tensor, offsets, 0, None, offs)

    import torch.distributed._symmetric_memory as symm_mem

    _enable_symm_mem(group_name)
    tensor = symm_mem.empty(nbytes_total, dtype=torch.uint8, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    ptrs = [
        handle.get_buffer(peer, (nbytes_total,), torch.uint8, storage_offset=0).data_ptr()
        for peer in range(world_size)
    ]
    base = ptrs[rank]
    offsets = torch.tensor([p - base for p in ptrs], dtype=torch.int64, device=device)
    mc_base = int(getattr(handle, "multicast_ptr", 0) or 0)
    tensor.zero_()
    return SymmetricArena(
        pos, phases, rank, world_size, tensor, offsets, mc_base, handle, offs
    )

"""Byte-addressed peer buffer device view (payload counterpart of EventSet)."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8
from cutlass.cute.typing import AddressSpace


@dataclass(frozen=True)
class PeerBuffer:
    """base: rank-local payload buffer. offsets[r] = peer_r_base - my_base.
    mc_base: NVLS multicast base address (0 when unavailable)."""

    base: cute.Pointer
    offsets: cute.Pointer
    my_rank: Int32
    mc_base: Int64

    @cute.jit
    def local_addr(self, byte_off: Int64) -> Int64:
        return self.base.toint() + byte_off

    @cute.jit
    def peer_addr(self, byte_off: Int64, dst_rank: Int32) -> Int64:
        off = cute.arch.load(self.offsets + dst_rank, Int64, sem="relaxed", scope="gpu")
        return self.base.toint() + byte_off + off

    @cute.jit
    def peer_ptr(self, byte_off: Int64, dst_rank: Int32, dtype, align: int = 16) -> cute.Pointer:
        return cute.make_ptr(
            dtype, self.peer_addr(byte_off, dst_rank), AddressSpace.gmem, assumed_align=align
        )

    @cute.jit
    def mc_ptr(self, byte_off: Int64, dtype, align: int = 16) -> cute.Pointer:
        return cute.make_ptr(
            dtype, self.mc_base + byte_off, AddressSpace.gmem, assumed_align=align
        )

    @cute.jit
    def has_mc(self) -> cutlass.Boolean:
        return self.mc_base != 0

# Gotchas

实现过程中已知的坑与设计约束，按主题分类。每条注明来源。

## 跨卡原子操作与事件系统（PTX 指令级分析结论）

| PTX | 完成语义 | 延迟（实测，B200 NVLink） | 用途 |
|---|---|---|---|
| `red.release.sys.global.add.u64 [peer], 1` | posted（无返回） | 单程传播 ~1.4µs 后远端可见，发射端≈0 | **跨卡 notify 唯一选择** |
| `ld.acquire/relaxed.sys.global.u64 [local]` | 每次轮询一次读 | ~0.2µs（L2）；acquire 与 relaxed 实测无差 | 本地自旋等共享 cell |
| `red.release.gpu.global.add.u64` | posted | CTA↔CTA 单程 ~1.1µs（含检测+barrier） | 本 rank 内 notify |
| `atom.acq_rel.sys.global.add.u64`（带返回） | **round-trip** | **~2.8µs 全往返** | 关键路径上禁止 |
| `fence.acq_rel.sys` | 栅栏 | **~1.7µs/次**（waitfix 实测：有 fence 1888ns vs 无 192ns） | 默认 wait 里的大头，见规则 6 |
| `multimem.red.release.sys.global.add.s32` | posted，交换机侧 | 未测（需 NVLS） | fan-in 事件首选 |
| `red.async.release.sys.global.add` + mbarrier | 挂 mbarrier | 不占发射端 | 延迟信号 |

**端到端实测（notify→wait 单程，pingpong RTT/2，B200:2 NVLink）**：带 fence 4.3µs；**flag-only（免 fence）2.85µs**；本地 gpu scope 1.1µs。预算：flag-only ≤3µs、本地 ≤1.5µs 已验收（tests/bench_notify_latency.py）。

1. **通知侧永远 `red`（posted），不用 `atom`（round-trip）**。tirx 的"最后到达者触发 push"（`old+1==target` 判断）在跨卡场景是一次 2-4µs 往返。**触发检测挪到消费侧**：consumer 自旋等 `cell >= target`，等到的 CTA 负责 push 后继，producer 只发 posted red。单卡内用 atom 无所谓（gpu scope 往返 ~0.4µs）。
2. **fan-in 事件（候选到达、LSE 到达等 cp 个 producer 的）优先 `multimem.red`**：每 rank 一条多播原子，交换机广播到所有 rank 的本地 cell，到达检测变成一次本地自旋。限制：**只支持 32-bit**（u64 相位编码需拆两个 32-bit cell 或压缩相位位宽），且需要 NVLS（NVSwitch）硬件。
3. **数据+标志顺序链**：TMA 推数据到 peer 是 async proxy，标志 red 之前必须 `cp.async.bulk.wait_group 0` + `fence.proxy.async`（flashinfer `gemm_allreduce_two_shot.py:1347` 踩过，用 `fence_proxy("alias")`）。`wait_group` 暴露 TMA 完成延迟 ~1-2µs → 推数据任务设计为 commit 后先干别的再 wait，禁止 commit 完立刻 wait。
4. **tirx 的远端 notify 用 `atom.release.gpu`（gpu scope）**（`tirx-kernels/.../utils.py:207-225`）——跨卡可见性不保证，我们跨卡一律 sys scope。tirx 自己的 megakernel MoE 是单卡的，该路径从未被验证过。
5. **notify 是"每调用点 +1"**：全 CTA 无守卫调用会 +128，单调计数器直接超射、所有 consumer 的 target 全错。CTA 级 API 内部守卫 thread0；warp 特化场景用未守卫的 `_1t` 变体，调用方保证单一发射者（events/core.py）。
6. **`fence.acq_rel.sys` 一次 ~1.7µs，是 wait 固定开销的 90%**（B200 实测：waitfix 有 fence 1888ns vs 无 fence 192ns）。因此生产 wait 用 **flag-only（`fence_after=False`）+ payload 侧自带排序**（payload 读用 scoped load / TMA acquire 链）；带 fence 的 wait 只给"flag 后面紧跟普通数据读"的场景（如测试的 payload 校验）。跨卡 notify→wait 因此 4.3µs→2.85µs。
7. **延迟基准必须用全新 cell 跑每一轮**：单调计数器饱和后 wait 退化成空转，测出来的是假数据（我们因此误报过 1.7µs 的"平坦曲线"，实际只有第一轮是真实的）。同理跨 GPU `%globaltimer` **不同步**（实测差 1.8e18 ns 的恒定偏移），不能用来测单程延迟，只能 RTT/2。
8. **cutlass `utils.distributed.multimem_st_*` 的寄存器参数收的是裸 MLIR Value，不是 Int32 Numeric**——官方示例里是 `multimem_ld_reduce` 的直接产出（`llvm.extractvalue` 裸值）。自己构造的值要 `Int32(v).ir_value()` 再传入，否则报 `Operand N must be a Value`（错误不指向真实出错行，先怀疑最近加的 asm 调用）。返回值同样裸，比较前 `Int32(r)` 包回来。
9. **Modal B200:2 的 NVLS multicast 可用**：torch symm_mem `handle.multicast_ptr` 非零（alloc_arena 里 `getattr(handle, "multicast_ptr", 0)` 取）。`multimem.st` 一次写全员落盘、`multimem.ld_reduce` 交换机侧 fp32 求和均已实测通过（phase03）。

## CuTe DSL / cutlass 平台限制

8. **CuTe DSL 无 NVSHMEM device 端**（cutlass `examples/python/CuTeDSL/cute/blackwell/kernel/distributed/README.md` 明示）。`nvshmem_ptr(addr, pe)` 只是地址翻译，等价物 = host 预算 per-rank offset 表（128B `grid_constant` 传入）+ device 端 `local + offsets[rank]` + `cute.make_ptr`。参考：`flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/src/sym_buffer.py:56-132`。
9. **CUDA 12/13 `nvvm.atomicrmw` MLIR 签名不兼容**，需要 shim（`flashinfer/cute_dsl/gemm_allreduce_two_shot.py:32-71`）。
10. **cutlass 4.3.1 缺 acquire-order CAS**，spin lock 要 workaround（同文件 `:129-133`，cutlass issue #2845）。
11. **跨卡自旋 barrier 要求 persistent grid 全 SM 共驻**，grid 必须按 `max_active_clusters` 限制，否则死锁（cutlass distributed 示例与 cutedsl_megamoe 均如此）。
12. **multimem 指令需要 NVLS 硬件**（NVL8/GB200 NVL72），raw peer VA FIFO 方案需要 MNNVL 统一地址空间（GB200）。

## CUDA graph / 内存管理

13. **NCCL host collective 在 graph capture 下死锁**（flashinfer `shim/comm.py:55-69`）；进 graph 的通信必须是静态 buffer + 设备侧 kernel。
14. **graph replay 无法改变 host 参数** → 跨 step 状态（相位、step 计数）必须设备侧维护；sense-reversing/单调相位编码的另一个动机是 ncu replay 无法快照跨设备 flag（cutedsl_megamoe `token_comm.py:1785-1797`）。
15. **vLLM 的 a2a buffer 必须来自 graph 私有池**，eager 分配会污染已捕获 graph（`vllm/v1/attention/ops/dcp_alltoall.py:111-129`）。
16. **NVSHMEM 对称张量无 GC**，要手动 free（flashinfer 测试代码注释）。

## 参考实现中已确认的 bug/缺陷（对照数值时注意）

17. **vLLM trtllm-gen 在 DCP + q_len>1 下因果掩码错误**：end-aligned mask 在交错分片的本地 KV 上，spec token i 会少看至多 `(dcp-1)(q_len-1-i)` 个 KV（含自身）（`vllm/v1/attention/backends/flashinfer.py:774-784` 注释）。我们的 kernel 必须 per-token 本地化 seq 上界。
18. **PR #44573 C4A decode 压缩索引映射非 DCP 感知**（`tmp/vllm-pr44573/.../cache_utils.py:517-565` 用非交错 block 映射，其余 kernel 全部用虚拟块布局）——C4A 层数值可能本来就是错的，gsm8k 短上下文测不出来。数值对照不一致时先查它。
19. **PR #44573 DSpark 非因果 SWA kernel 无 DCP 感知且无 gate**（`sparse_swa.py:773-834`）。
20. **vLLM DCP+prefill 的 metadata 构建有 GPU sync**（`indexer.py:801-802` 对设备张量 `.item()`，每 prefill chunk 2 次）。
21. **sglang CUDA 上 DCP+投机解码只允许 Kimi Linear+DSPARK**（`server_args.py:3790-3818`），DeepSeek+MTP+DCP 直接 raise——跨框架对照时别踩。

# Event 系统（etensor）spec

Phase 0.1 交付物。对标 tirx-kernels 的事件系统（`tmp/tirx-kernels/tirx_kernels/megakernel/utils/{static,dynamic}_scheduler.py`），扩展跨卡能力。整个 megakernel 的算子间依赖、流水重叠、跨卡同步都建立在这一套原语上。

> **实现状态**：**Phase 0.1 验收通过**（Modal B200 / B200:2，`scripts/modal_app.py`）。CPU 校验器 50 随机 DAG + 8 错误注入；单卡 3 种子随机 DAG（512 任务/~460 事件/~2600 边）执行序全部合法；双卡 ping-pong/fan-in/fan-out 1000 相位 + payload release/acquire 链校验通过。延迟实测：本地 1.1µs、跨卡 flag-only 2.85µs、带 fence 4.3µs（fence.acq_rel.sys 一次 ~1.7µs——生产 wait 默认 flag-only，payload 排序下沉到消费侧，见 gotchas #6/#7）。代码 `mega_dsa_cp/events/{ptx,core,table,validator,symm}.py`；测试 `tests/`；一键 `scripts/run_phase01.sh`（本地）或 `modal run scripts/modal_app.py::{single_gpu,dual_gpu}`。

## 1. 设计目标

- tile 粒度的生产者-消费者依赖（graph 的 kernel 级栅栏的替代品）
- 单阶段 notify 起步（静态调度器）；两阶段 notify（**push 可以早，读不能早**）保留给 Phase 3 动态调度器
- 同一套语义覆盖本 rank（gpu scope）与跨卡（sys scope）事件
- persistent kernel 跨 step（graph replay / 逐层逐 step 重 launch）复用，无清零竞态（算子边界=单层，见 overview.md 定位）
- CUDA graph 可捕获（无 host 依赖值，拓扑 capture 时锁死）

## 2. 单元布局

- 事件 = 对称内存中的 **64-bit 计数单元**（全部事件统一放 symm-mem，本地/跨卡不区分，简化布局）
- 逻辑索引由 host 静态分配：`event_id = f(layer, event_type, tile_coord)`，事件表按层型（SWA / C128A / C4A）× 固定 shape 声明
- 位段：`cell = (finished_count << 32) | dispatched_count`
- 每个事件声明静态 `producer_count = P`（由图拓扑推出，CPU 检查器校验）

## 3. 计数语义与相位编码

单调递增 u64，**不做 tirx 式"P×(base+1) 递减 + INIT_ETENSOR 初始化 + 竞态恢复"**（persistent 复用下有清零竞态，且 init 与 notify 乱序恢复回路每次 notify 多付一次原子操作）：

- **单阶段 notify（静态调度器模式，Phase 0-2 只用这个）**：`notify(cell)` = `red.release.{scope}.add.u64 [cell], 1` —— tile 体完成后调用；read-wait 条件 = `cell >= (phase+1) * P`
- **两阶段（保留给 Phase 3 动态调度器）**：`pre_notify` 加 1 到低 32 位（`add 1`），`notify` 加 1 到高 32 位（`add 1<<32`）；位段 `cell = (finished << 32) | dispatched`。push-wait 条件 = `lo32 >= (phase+1)*P`，read-wait 条件 = `hi32 >= (phase+1)*P`。最后到达者触发 push：atomicAdd 返回 old，`old+1 == target` 的线程负责 enqueue 后继（对应 tirx 的 `is_triggered`）
- phase k（第 k 次使用）：事件按 (layer, coord) 分配，**每 decode step 内 one-shot**（无需步内相位）；跨 step 相位 = 设备侧 step 计数器（每 replay 由指定任务 +1，wait 目标从它算出）
- scope 选择：`gpu` = 只被本 rank CTA 等待的事件；`sys` = 会被远端 rank notify 的事件
- 跨卡 notify：对 peer 的事件地址 `red.async.release.sys`（peer 地址 = symm base + offsets[rank]，offset 表 host 预计算经 grid_constant 传入）。注意 tirx 的远端 notify 用的是 gpu scope（`utils.py:207-225`），跨卡可见性存疑，必须用 sys

### 相位与 rebase

- CUDA graph replay 时 host 无法改参数 → 设备侧维护 **step 计数器**（每 replay 由指定任务 +1），wait 目标 = (step+1) × P，各 CTA 在 kernel 入口读一次缓存到 SMEM
- 事件区只在**首次分配时清零一次**（symm-mem 分配后 memset），之后单调累积，每 decode step 无需清零（这是相对 tirx 每步清零的主要简化）
- u64 下溢出实际不可达（2^32 次 notify），**不做 rebase**

## 4. 原语伪码

```
notify(cell, scope):                  # 单阶段（静态调度器）
    red.release.{gpu|sys}.global.add.u64 [cell], 1

pre_notify(cell, scope):              # 两阶段（动态调度器，Phase 3）
    red.release.{gpu|sys}.global.add.u64 [cell], 1
notify_2p(cell, scope):
    red.release.{gpu|sys}.global.add.u64 [cell], 0x1_0000_0000

wait_read(cell, target, scope):       # 单阶段 target = (step+1) * P
    while ld.acquire.{gpu|sys}.global.u64(cell) < target:
        nanosleep(backoff)

wait_push(cell, target, scope):       # 两阶段专用
    while (ld.acquire.{gpu|sys}.global.u64(cell) & 0xFFFFFFFF) < target:
        nanosleep(backoff)
```

- CTA 级 wait（借自 tirx）：全员 `ld.acquire` + `syncthreads_and(cond)` 一次 barrier 聚合退出；warp 级 = lane0 读 + `any_sync` 广播；warp 级以上 notify 由 elect_one 执行
- backoff：初始 40ns，指数上限 ~1µs（跨卡事件上限 ~4µs，避免自旋挤占 NVLink）
- 跨卡 barrier（step 边界等全局同步点）：专用 fan-in 事件（每 rank 一个 notify）+ 相位阈值，不做递减归零
- notify 代码生成：任务图固定，每个任务的 notify 列表静态已知，直接生成直线代码（不用 tirx 的 func_notify 通用映射机制）

## 5. 事件表（每层型初稿，随任务图演进）

| 事件 | producer(s) | consumer(s) | scope |
|---|---|---|---|
| `q_ready[head_chunk]` | W_UK bmm tile ×4 | attention tile | gpu |
| `q_peer_arrived[rank][head_chunk]` | q 推送任务（远端） | attention tile | sys |
| `logits_done[kv_chunk]` | logits tile ×N_chunk | 本地 topk tile | gpu |
| `cand_ready` | 本地 topk tile | 候选推送任务 | gpu |
| `cand_arrived[rank]` | 候选推送（远端 ×cp） | 全局合并 tile | sys |
| `topk_done` | 全局合并 tile | remap tile | gpu |
| `attn_done[head_chunk]` | attention tile ×4 | LSE 合并任务 | gpu |
| `lse_arrived[rank]` | LSE 推送（远端 ×cp） | 输出修正 tile | sys |
| `out_merged[head_chunk]` | 输出修正/RS 任务 | W_UV tile | gpu |
| `kv_written` | cache 写 tile | （下一 step 的 logits） | gpu |
| `cmp_stats_arrived[rank]` | compressor stats 推送（远端 ×cp） | compressor finalize tile | sys |
| `layer_done[layer]` | 每层收尾任务 | 下一层首个任务 / step 边界 | gpu |

规则：每个事件恰有一种 consumer 等待模式；fan-in 事件 P = producer 数；跨卡事件在所有 rank 上对称声明（collective 一致性）。

## 6. CPU 静态检查器（spec.validate 对应物）

host 生成任务图后、launch 前校验：

1. wait/notify 边构成 **DAG**（无环；图是一层的任务图，跨 step 一致性靠单调相位而非图内边）
2. 每个事件的 `producer_count` 与实际 notify 边数一致；fan-in/fan-out 声明匹配
3. 每个事件的 phase 目标在每次使用时可静态计算（shape padding 范围内）
4. 跨卡事件在所有 rank 上拓扑对称（各 rank 生成的图做哈希对比）
5. 无 host 依赖值（graph capture 兼容）；事件区总大小 ≤ 预算（初定 16MB symm-mem）

## 7. 验收测试

1. CPU 检查器：合法图通过；注入环 / 计数不匹配 / 非对称跨卡图均能报出
2. 单卡等价性：合成任务图（随机依赖 DAG，~10K 任务）乱序执行结果与串行执行 bit 一致
3. 双卡原语：ping-pong（单事件来回 10K 次无丢相位）、fan-in（7 producer → 1 consumer）、fan-out
4. **延迟基准**（决定流水线粒度下限的物理常数，先于一切测出）：
   - 单卡 notify→wait 观测延迟（目标 < 200ns）
   - 跨卡 notify→wait 观测延迟（目标 ≤ 3µs；NVL72 域内）
   - 与 NCCL 同语义同步的开销对比（支撑收益模型中"核内 ~3-5µs vs NCCL ~10-15µs"的假设）
5. 压力：单 kernel 实例（一层）连续 replay 1000 次（= 61 层 × ~16 step 的调用量），无死锁、无相位错乱（配合 0.4 空壳 pipeline）

## 8. 与 tirx 的对应关系（逐行核对源码后的取舍）

| tirx | 本设计 | 取舍 |
|---|---|---|
| int32 单元 `P×(base+1)` 递减编码，INIT_ETENSOR 任务初始化 | u64 单调相位计数，首次分配清零一次 | **另起**：免 init 任务、免竞态恢复回路、免每步清零 |
| notify 抢在 init 前的恢复回路（old≤0 自旋补减） | 不需要（无 init 竞态） | **另起** |
| 静态调度器单阶段 `-(base+1)` | 单阶段 `+1`，wait `>= (step+1)*P` | 语义同构 |
| 动态调度器两阶段 + `is_triggered`（old%base==1 → 最后到达者 push） | 两阶段（高低 32 位）+ `old+1==target` 触发 push，**Phase 3 动态调度才启用** | **参考**：这是 tirx 事件系统真正的核心 IP |
| CTA 级 wait：`syncthreads_and(state==0)`；warp 级 lane0+`any_sync`；nanosleep(40) | 相同 | **直接借** |
| wait/notify scope 枚举（thread/warp/warpgroup/CTA） | 相同分类，但 notify 列表静态生成直线代码，不用 func_notify 通用映射 | **简化** |
| 远端 notify `atom.release.gpu`（`utils.py:207`） | `red.async.release.sys` | **修正**：跨卡必须 sys scope（tirx 该路径未经多卡验证） |
| wait 条件 == 0 | wait 条件 >= 阈值 | 语义同构 |

源码位置：`tmp/tirx-kernels/tirx_kernels/megakernel/utils/{static_scheduler.py:29-76, dynamic_scheduler.py:50-104, base.py:441-486, utils.py:207-307}`

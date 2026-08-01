# mega-dsa-cp 设计总览

DeepSeek V4 DSA（DeepSeek Sparse Attention）decode 的 megakernel 实现：把 indexer logits → topk → sparse MLA attention → CP 通信融合为一个 persistent kernel，面向长上下文 + 投机解码 + CP（context parallel）场景。

## 定位与边界

- 这是 megakernel 在 CP 上的**实验**，不是生产路径
- 目标硬件：SM100/SM103（B200/GB200），NVLink 超节点
- 部署语境：AFD（Attention-FFN Disaggregation）分离，MoE 资源单独供给，**只优化 attention 段**
- **算子边界 = 一个 decode step 的一层**：kernel 处理单层 DSA attention + CP 通信，runner 每 step 调 61 次（或 CUDA graph 内 61 节点 replay）。不做 61 层全跑——那会把层间 FFN 的 61 次 AFD 往返吞进来，越界。层型（C128A/C4A）在 launch 时选择，kernel 内无层循环
- 形状假设：投机解码常开，q_len = 1 + n_spec（MTP-1 → 2；DSpark block 5-7 → 6-8）；decode 用均匀 batch（固定 shape）
- 只考虑长上下文（S ≥ 64K）；短上下文不需要 CP，直接 DP attention，不在参数空间内
- S 与 step 计数是**设备侧标量**（graph replay 间 host 不可写）：事件用 u64 单调相位计数（target = arity×(phase+1)），replay/重 launch 零清零

## 调研结论（参考实现现状）

| 系统 | DSA decode CP 现状 |
|---|---|
| vLLM | V3.2 走通用 DCP（KV+indexer 全分片，CuteDSL 层级 topk，flashinfer 后端限定，2-3 次 NCCL/层）；V4 无任何 CP（compress_ratio>1 硬拒 DCP，PCP 半成品） |
| vLLM PR #44573 | V4 DCP decode 未合并实现（gatherQ + FlashMLA + A2A）。实测 DCP4 比 TP4 **慢 35-39%**（<128K 上下文），128K 才转正。每层 2-7 次 eager NCCL，每步 150-300+ 次；C4A decode 索引映射疑似有 bug；DSpark 不安全 |
| sglang | DSA CP 是独立开关但**只管 prefill**（KV 全复制，decode 各 rank 冗余计算，零 CP 通信）；普通 DCP 不接 DSA 后端，且 CUDA 上 DCP+投机只允许 Kimi Linear |
| vllm-ascend | 三种形态：token 分片 DSA-CP（KV 复制，计算÷cp，权重复制）；SFA DCP（KV 分片 + indexer 复制 + remap）；fused lightning_indexer（logits+topk 单 kernel）已产品化；MTP draft 步间 topk 复用（skip_topk/IndexCache） |
| flashinfer PR #3943 | CuTe DSL HCA kernel（V4 sparse MLA decode，SWA 128 窗口 + 128:1 压缩双池，fp8，128 head，512 latent，2-CTA cluster，warp 专用化，persistent 可选，q_len>1 支持）；**无跨卡通信**，topk 外部输入 |
| flashinfer cutedsl_megamoe | CuTe DSL 多卡 megakernel 完整范式：symm-mem + peer offset 映射 + sense-reversing barrier + TMA pull/red.sys push |
| tirx-kernels | megakernel DSL（TIRx/TVM）：persistent kernel + packed task 队列（静态/动态 MPMC）+ etensor 两阶段 notify；bs1 收益 1.66x；先手写后 DSL 的工程路线 |

## CP 语义选择

**全 KV 分片（vLLM DCP 语义）**：indexer cache + MLA KV 都按 token 交错分片，层级 topk（本地 topk → 候选交换 → 全局合并），q head 维交换，LSE 合并输出。

**拓扑静态、标量动态**：DSA decode 的任务图拓扑与 S 无关——topk K 是固定超参（S 依赖被截断在 indexer 段），SWA 窗口 128 固定，attention 本体（K+128 候选）与 q 路 bmm 天然固定 shape；随 S 变的只有 indexer logits tile 数与 topk 归并深度，是**标量动态**（trip count），不是拓扑动态。因此静态调度成立：队列按 S_bucket 枚举 tile 对，任务内循环上界读设备侧 seq_len 标量，越界槽位谓词跳过但照发 notify（事件图完整）；ragged batch 的零浪费精确调度归 Phase 3 动态调度器。对比：tirx 上动态调度是因为 MoE expert 计数是数据决定的拓扑动态，我们没有这个结构。

理由（decode 瓶颈结构决定）：
- 长上下文 decode 的瓶颈是 indexer 计算（∝ S×B×q_len）+ 通信 + launch 开销；短上下文是权重带宽 + launch 开销，**不是激活计算**
- token 分片（Ascend DSA-CP）会复制 attention 权重（权重流量 ×cp），decode 中小 batch 下是负优化，只适合 prefill/超大 batch
- KV 分片保持权重 TP 切分不变，计算和显存同时 ÷cp
- 二维 (token×KV) 分解留接口不实现（cp≥16 时控制候选交换载荷用）

## megakernel vs 最优 graph 实现（收益口径）

graph 已能拿到（不计入 megakernel 收益）：CPU launch 消除、独立算子多流 overlap、自定义融合 kernel（如 Ascend lightning_indexer）、NCCL 进 graph。

megakernel 独有：
1. **依赖算子间 tile 级流水**（graph 依赖粒度是整个 kernel；etensor pre_notify 使后继任务在 producer 完成前即可派发）
2. **通信与计算融合 + 分块流水**（NCCL 是整体栅栏，固定 10-15µs/次不可流水；核内 push/pull ~3-5µs 且可被计算遮盖）
3. **数据依赖的动态执行**（变长 seq 精确 tile 调度 vs max padding；skip_topk；draft 循环吸收）
4. **持久化**（一次启动，元数据/调度器/流水线状态在本层执行期内常驻；跨 step 靠单调相位免清零 replay）

关键 overlap 设计（每层，KV 分片 + spec decode）：
- q 路（q_b + W_UK bmm）∥ indexer 路 ∥ KV 写入，三分支独立
- q 交换按 head-chunk 流式推送，藏进 indexer logits
- 候选交换按 KV chunk / spec token 行分块流水
- **spec token 链级流水**：logits 共享 K tile 一把算完（multi-atom），之后 q_len 条独立链（topk→候选交换→合并→attention→LSE 合并）交错，通信全隐藏
- LSE 合并按 head-chunk 切分流水
- （可选）W_UV 提到合并前：RS 载荷 512→128 维/head，代价 W_UV 权重复制

估算（S=128K, B=32, q=2, cp=8，attention 段/step）：最优 graph ~4.5-6ms → megakernel ~1.8ms（**2.5-3x**）；通信与序列化残差不随 S 缩短，短上下文相对收益更大；cp 越大 graph 通信地板越硬，差距继续拉大。

## 通信清单（0.3 冻结，每层每 step 每 rank）

| 数据 | 量级 | 传输 |
|---|---|---|
| topk 候选 (logit, 条目id) 对，**压缩条目/页粒度** | ~128KB | multimem.st allgather（无 NVLS → cp-1 次 TMA push） |
| LSE 标量 | ~32KB | multimem.st allgather |
| LSE partials（**fp32 直通**，不做降级转换；W_UV-before-merge 后 128/head） | ~4MB（最大头） | multimem.ld_reduce 归约（fallback TMA pull + 本地加）；按 head 分 chunk 流水 |
| q | **不交换**（hidden states 在 attention 侧复制，#44573 gatherQ 是反例） | — |
| 新 token KV/indexer 写入 | KB 级 | 纯本地（owner=token%cp，本地算本地写） |

关键决策：跨卡**零 indexed gather**——选中 KV 的覆盖靠"attention 留本地 + LSE 合并"（V3.2 DCP 语义），index 只作为稠密候选数组的内容传输。attention 本地读取（HCA 范式，flashinfer hca_fp8.py）：双池（SWA 窗 + 压缩池）page table + TMA tile load，稠密枚举 + per-query 有效长度掩码（`sparse_mla_topk_lens`）；选择在选择物化为页索引（sglang 页粒度，64 对齐 page_bits 编码）或位置推导（C128A 零 topk 通信）。

## 实验协议（固定 shape）

每个配置点固定四元组 **(S, B, q_len, cp)**，均匀 decode batch：
- 核心网格：S ∈ {64K, 128K, 256K} × B ∈ {8, 32, 128} × q_len ∈ {2, 8} × cp ∈ {4, 8}；外加 S=8K 定位 crossover
- 四个对照：① vLLM V4 TP（开 MTP，及格线）② PR #44573 DCP（同语义实现）③ megakernel cp=1（隔离融合收益）④ megakernel cp=k
- 指标：attention 段 step 时间（µs/step）；五段分解（logits/topk/attention/通信暴露/调度残差）；crossover 曲线（megakernel DCP 从哪个 S 开始打过 TP，目标压到 16-32K 以下）；cp 扩展性曲线（固定 128K/32/2，cp ∈ {2,4,8,16}）
- 正确性：megakernel cp=k vs **vLLM V4 TP（cp=1，同算法）**，固定 seed 输入，逐层 attention out 对数值 + topk 索引集合一致；层级 topk 合并单测对 vLLM CuteDSL 实现；#44573 的 C4A 数值不可信（疑似 bug），不一致先查它

## 分阶段计划

图拓扑（任务类型、通信边、事件表、buffer 布局）从骨架阶段**冻结**；cp=1 与 cp=k 是同一幅图的两种执行，通信任务在 cp=1 下退化但仍在图中，不存在后期追加边。

- **Phase 0 骨架**：0.1 event 系统（见 event-system.md，已验收）→ 0.2 静态队列调度器（**已验收**：`mega_dsa_cp/schedule/{codegen,device}.py`——codegen 把校验过的单层 DAG 拓扑序发牌成每 CTA packed 队列（header + 3 坐标 + 内联 wait/notify 边，wait≤63/notify≤63），设备端 smem 暂存 + cursor 遍历 + scope 分派 wait + 单发射者 notify + 动态上界标量读取；CPU 30 随机 DAG + 多 rank 对称性过，单卡 512 任务×2 种子×2 相位过，双卡跨 rank DAG×2 相位过）→ 0.3 通信原语+buffer 布局（**已验收**：`mega_dsa_cp/comm/{primitives,device,buffers}.py`——bulk S2G push（`cp.async.bulk` 1D + commit/wait + `fence.proxy.async.global` + notify 链）、`multimem.st` allgather、`multimem.ld_reduce` fp32 交换机归约、SymmetricArena 命名区域×相位轮换；双卡 2 相位全过，NVLS multicast 在 Modal B200:2 可用）→ 0.4 空壳 pipeline cp=2 端到端（**骨架已冻结**：`tests/test_skeleton_cp2.py`——30 任务/rank、14 事件的全拓扑过调度器；候选 128KB multimem.st allgather 校验、partials 4MB（16×256KB）smem→bulk push 到 owner inbox 双 rank 校验、2 相位单调复用全过；热态 wall 566µs/层（stub 算子 + 真实通信量），冷启动 6.9ms）**Phase 0 完成**
- **Phase 1 计算 tile**（图不变，cp=1 退化执行下对数值）：1.1 logits+topk 融合 → 1.2 attention（HCA 改造）→ 1.3 compressor → 1.4 通信任务实现 → 1.5 GEMM tiles
- **Phase 2 跨卡执行**：cp=2→4→8，数值双参照（自身 cp=1 + vLLM TP），性能实验矩阵
- **Phase 3（可选）**：DSL 化（tirx 方法论：队列字节等价+数值等价）、动态 MPMC 调度、2D 分解

## 运行环境（Modal）

本机只做编辑与 CPU 级校验；GPU 执行走 Modal（模式与 svdquant-kernels 一致：源码 `add_local_dir(copy=False)` 挂载、设备端 JIT，无本地构建）。入口 `scripts/modal_app.py`：`modal run scripts/modal_app.py::{smoke,single_gpu,dual_gpu,phase02_single,phase02_dual,phase03_dual,phase04_dual}`，日志 `log/<task>.log`。镜像 = debian_slim py3.12 + torch 2.11 cu130 + nvidia-cutlass-dsl + cuda-python。注意：Modal 屏蔽计数器级 profiling（ncu、`nsys --gpu-metrics-device` 不可用；torch.profiler / nsys 时间线可用）；每条命令设硬超时（kernel 秒级、编译 ~1 分钟，超时即死锁）；`retries=0` 防整套件重跑烧钱。双卡测试用 `gpu="B200:2"`（单节点 NVLink P2P + torch symm_mem rendezvous over nccl group）。

## 参考代码（tmp/）

文档：`docs/overview.md`（本文）、`docs/event-system.md`（Phase 0.1 spec）、`docs/gotchas.md`（已知坑与设计约束）

- `tmp/flashinfer`（分支 pr-3943）：HCA kernel，`flashinfer/cute_dsl/attention/dsa/`
- `tmp/flashinfer-pr3381`：DSA compress+norm+rope CuTe DSL kernel
- `tmp/flashinfer/moe_ep/kernel_src/cutedsl_megamoe/`：多卡 megakernel 范式
- `tmp/tirx-kernels`（分支 agent/megakernel-dsl）：megakernel DSL、etensor、调度器
- `tmp/vllm`：mainline vLLM（V3.2 DCP、V4 模型）
- `tmp/vllm-pr44573`：V4 DCP 基线实现
- `tmp/vllm-ascend`：DSA-CP/SFA DCP/fused indexer
- `tmp/sglang`、`tmp/cutlass`

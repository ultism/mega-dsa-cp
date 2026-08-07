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
| TRT-LLM blog26（V4 on Blackwell） | V4 三层型定义：ratio 0=SWA-only、4=CSA（**entry 级 top-k + gather**，topk=512 Flash/1024 Pro）、128=HCA（dense 全量、无 indexer）；CSA 层双压缩流（attention KV + indexer-K 同坐标系）；**MXFP4 index-K 是 TRT-LLM 默认**（"FP4 approximately halves the Indexer-K data payload"——佐证我们 FP4 决策）；压缩池页按 raw-token 坐标分配（tokens_per_block/r entry/页）；top-k 优化：radix/insertion 分派 + GVR 时序复用（相邻步选择重叠，1.4-2.17×）；compressor = softmax-gated pooling，CSA 两个相邻 4-token 组 8-token 感受野；attn_sink checkpoint 自带 |
| flashinfer PR #3943 | CuTe DSL HCA kernel（V4 sparse MLA decode，SWA 128 窗口 + 压缩池双池，fp8，128 head，512 latent，2-CTA cluster，16 warp 专用化，persistent 可选，q_len>1 支持）；**页粒度 dense-prefix 枚举**（`sparse_topk_lens` 前缀掩码，无 scattered topk）；**无跨卡通信**，选择外部物化为页表输入 |
| sglang #26209/#27059/#30546 | FP4（MXFP4）indexer 数据：index-K 132B→68B/token/层（-48%，+9.9% KV 容量），logits kernel 1.49-1.53×（SM100 L=8K/32K），E2E +6-8%，精度 GSM8K -1.5pt / GPQA ~0；sglang 做成 opt-in 默认关（产品保守）；kernel 在 DeepGEMM（`fp8_fp4_paged_mqa_logits`） |
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

## C4A 选择粒度决策：entry 粒度（与官方一致）+ HCA fork 双模式消费

三方调研互证（2026-08-03 确认）：
- **sglang/vllm 的 C4A 是 entry 粒度**：indexer 在 S/4 个 C4 entry 打分上 topk-512/1024（sglang `_topk_transform_512_vectorized` 输出 `physical_page << page_bits | offset` 的 entry 级物理索引；vllm `topk_indices_buffer` 同），注意力走 **FlashMLA sparse 逐 entry gather**（`flash_mla_with_kvcache` + `extra_indices`）
- **C128A 无学习选择**：dense 全量压缩条目（vllm `build_c128a_topk_metadata` 从 positions 推导槽位列表 + 计数，**indexer 不参与 C128A 层**——这些层没有 logits/topk tile）；sglang 同（`c128_page_indices` 位置推导）
- **flashinfer HCA（cute_dsl hca_fp8.py）是页粒度 dense-prefix 枚举**：压缩池按页表整页 TMA 装载，`sparse_topk_lens` 只做逻辑前缀掩码，选择必须以"页表列哪些页"物化

**决策：C4A 选择保持 entry 粒度（与官方语义完全一致），放弃页粒度作为 baseline。** V4 index_topk = 512（Flash）/1024（Pro）（sglang `configs/deepseek_v4.py:60`、TRT-LLM blog26；注意不是 V3.2 的 2048），cand 格式 = (logit fp32, entry_id u32)。理由：页粒度 CSA 选择是公开未验证语义（见下「风险确认」），研究 baseline 必须与 sglang/vllm/TRT-LLM 的 entry 级 top-k 对齐，否则 E2E 数值对拍失去意义；页粒度降级为后续 ablation（做完 exact baseline 后受控实验，量化 recall 差）。

attention 消费形态（保证 fork 计划不受回退影响）：
- **C128A + SWA-only 层：fork HCA 页枚举**——dense 全量，页枚举 = 精确语义，零风险
- **CSA 层：同一个 HCA fork 的 gather 模式**——cmp 流 `page_size_cmp=1`（每 entry 512B fp8，满足 TMA 128B 对齐；原版 `can_implement` 拒 page_size≤1 是其通用布局保守，我们放开），**cand merge 产出的 entry id 列表直接当 1-entry 页的页表**，无格式转换。代价是每 k-tile 128 次 512B TMA（vs 页版 8 次 8KB），字节数相同，损失的是空间局部性与 TMA 发射开销；`load_tma_qk_one_k_tile` 需按页 chunk 重构（k_idx 寄存器张量 128×4B 超预算，改 8 个一批流水）。若 TMA gather 效率不达标，备用方案是 FlashMLA sparse 式 LDGSTS gather（V3.2 路线）
- **通信**：候选回到 entry 级（见下表）；层级 topk 归并语义与官方完全一致（合并 entry 列表）

**风险确认（2026-08-03 调研，回退依据）：页粒度 CSA 选择是公开未验证语义。** ①flashinfer PR #3943 全文未讨论精度问题——它只把 page-aligned 当 ABI 门槛（`hca_sparse_indices_format="page-aligned"` 显式拒绝 arbitrary sparse token selections），entry 粒度的 TRTLLM-GEN 路径仍是默认，benchmark 只在同一 page-aligned 表上双后端对拍（max diff 0.0039），从未比较"页粒度选择 vs entry 粒度选择"；review 无人工讨论（全 bot）。②TRT-LLM blog26 明确写 CSA = entry 级 top-k + **gather**（"gathers selected CSA entries"、"Selected compressed positions are converted to token-level global addresses"），其 Top-K 优化（radix/insertion、GVR 时序复用）也全在 entry 坐标上。即所有生产路径（sglang/vllm/TRT-LLM）都是 entry 粒度。结论：baseline 必须 entry 粒度；页粒度作为后续 ablation 时，用 1.1 参考路径在 torch 级量化 recall（entry-topk vs 页 max-topk 的集合重合度 + attention out 偏差）。

影响：
- **1.1 改造点（1.1b）**：topk K 对齐 V4 官方（512/1024，堆容量从 2048 右尺寸化到 1024 释放 smem）；cand 格式 = (logit fp32, entry_id u32)，**entry 粒度不变**
- **attention tile（1.2）= fork `hca_fp8.py`**：CSA 层 cmp 流 gather 模式（page_size=1，entry id 列表即页表）；C128A 变体页表位置推导（dense 页枚举）；**第三种层型 SWA-only（ratio 0，Flash 前两层）= cmp 池为空的退化变体**（TRT-LLM blog26 三层型定义）
- **通信清单**：候选行回到 entry 粒度（见下表）

## indexer 精度决策：直接 FP4（MXFP4），不留 FP8 旋钮

打分路径（index-K cache + q_indexer + logits MMA）一步到位 MXFP4，不做 FP8/FP4 双模。理由：sf（scale factor）路径决定 tile 布局、smem 分区、MMA 类型与量化写路径，后期补加等于重做 logits/compressor 两块 tile；megakernel 的图与 buffer 布局冻结后可调旋钮少，实验算子直接锁定终态格式。sglang 默认 FP8 是产品保守（精度 -1.5pt GSM8K），实验路径不需要这个保守。

- **格式**：index-K 每 entry = 64B packed E2M1 + 4B（4×UE8M0 组 32 scale）= 68B（page=64 entry 布局不变：数据段后接 scale 段）；q_indexer 同样 FP4 + q_sf，**q_sf 由 kernel 内应用**，不折进 head-gate weights（weights 保持 fp32 直达 epilogue）
- **MMA**：mxf4 block-scaled UMMA（SM100 原生，K=128 全 K）；SF 走 TMEM 布局
- **写路径**：compressor tile 产出 index-K 时量化 E2M1 + UE8M0 ceil scale（对 DeepGEMM 量化器 bit-identical：`max(amax/6, 1e-4)`、过 bf16 舍入）
- **数值参照**：tile 级对 DeepGEMM `fp8_fp4_paged_mqa_logits`（同输入逐值）；E2E 仍对 FP8 参考实现做 topk 集合召回 + attention out 容差对比（FP4↔FP8 选择差异是预期的）
- **参考实现**：`~/svdquant-kernels/cute_kernels/gemm_w4a4/`（自有 W4A4 blockscale GEMM，含 TMEM/SF 布局验证）、cutlass `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/`（SM100/103 persistent blockscaled GEMM）+ `cutlass/utils/blockscaled_layout.py`；flashinfer `msa_ops/cute_dsl/proxy_score_fp4_sm12x.py`（注意力场景 FP4 MMA）；sglang `cutedsl_fp8_paged_mqa_logits.py`（pipeline 骨架）；DeepGEMM fp8_fp4（数值 oracle）

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
| topk 候选 (logit, entry_id) 对，**entry 粒度**（512/1024 entry/query，4-8KB/query） | ~256-512KB（B=32,q=2） | multimem.st allgather（无 NVLS → cp-1 次 TMA push） |
| LSE 标量 | ~32KB | multimem.st allgather |
| LSE partials（**fp32 直通**，不做降级转换；W_UV-before-merge 后 128/head） | ~4MB（最大头） | multimem.ld_reduce 归约（fallback TMA pull + 本地加）；按 head 分 chunk 流水 |
| q | **不交换**（hidden states 在 attention 侧复制，#44573 gatherQ 是反例） | — |
| 新 token KV/indexer 写入 | KB 级 | 纯本地（owner=token%cp，本地算本地写） |

关键决策：跨卡**零 indexed gather**——选中 KV 的覆盖靠"attention 留本地 + LSE 合并"（V3.2 DCP 语义），index 只作为稠密候选数组的内容传输。attention 本地读取（HCA 范式，flashinfer hca_fp8.py）：双池（SWA 窗 + 压缩池）page table + TMA tile load，稠密枚举 + per-query 有效长度掩码（`sparse_mla_topk_lens`）；选择物化为 **entry 级索引列表**（CSA：topk-512/1024 entry，merge 树产出，attention 以 page=1 gather 消费；C128A：dense 位置推导，零 topk 零 logits）。注：CSA 的 entry 粒度语义与 sglang/vllm/TRT-LLM 生产路径完全一致；页粒度仅作后续 ablation。

## 实验协议（固定 shape）

每个配置点固定四元组 **(S, B, q_len, cp)**，均匀 decode batch：
- 核心网格：S ∈ {64K, 128K, 256K} × B ∈ {8, 32, 128} × q_len ∈ {2, 8} × cp ∈ {4, 8}；外加 S=8K 定位 crossover
- 四个对照：① vLLM V4 TP（开 MTP，及格线）② PR #44573 DCP（同语义实现）③ megakernel cp=1（隔离融合收益）④ megakernel cp=k
- 指标：attention 段 step 时间（µs/step）；五段分解（logits/topk/attention/通信暴露/调度残差）；crossover 曲线（megakernel DCP 从哪个 S 开始打过 TP，目标压到 16-32K 以下）；cp 扩展性曲线（固定 128K/32/2，cp ∈ {2,4,8,16}）
- 正确性：megakernel cp=k vs **vLLM V4 TP（cp=1，同算法）**，固定 seed 输入，逐层 attention out 对数值 + topk 索引集合一致；层级 topk 合并单测对 vLLM CuteDSL 实现；#44573 的 C4A 数值不可信（疑似 bug），不一致先查它

## 分阶段计划

图拓扑（任务类型、通信边、事件表、buffer 布局）从骨架阶段**冻结**；cp=1 与 cp=k 是同一幅图的两种执行，通信任务在 cp=1 下退化但仍在图中，不存在后期追加边。

- **Phase 0 骨架**：0.1 event 系统（见 event-system.md，已验收）→ 0.2 静态队列调度器（**已验收**：`mega_dsa_cp/schedule/{codegen,device}.py`——codegen 把校验过的单层 DAG 拓扑序发牌成每 CTA packed 队列（header + 3 坐标 + 内联 wait/notify 边，wait≤63/notify≤63），设备端 smem 暂存 + cursor 遍历 + scope 分派 wait + 单发射者 notify + 动态上界标量读取；CPU 30 随机 DAG + 多 rank 对称性过，单卡 512 任务×2 种子×2 相位过，双卡跨 rank DAG×2 相位过）→ 0.3 通信原语+buffer 布局（**已验收**：`mega_dsa_cp/comm/{primitives,device,buffers}.py`——bulk S2G push（`cp.async.bulk` 1D + commit/wait + `fence.proxy.async.global` + notify 链）、`multimem.st` allgather、`multimem.ld_reduce` fp32 交换机归约、SymmetricArena 命名区域×相位轮换；双卡 2 相位全过，NVLS multicast 在 Modal B200:2 可用）→ 0.4 空壳 pipeline cp=2 端到端（**骨架已冻结**：`tests/test_skeleton_cp2.py`——30 任务/rank、14 事件的全拓扑过调度器；候选 128KB multimem.st allgather 校验、partials 4MB（16×256KB）smem→bulk push 到 owner inbox 双 rank 校验、2 相位单调复用全过；热态 wall 566µs/层（stub 算子 + 真实通信量），冷启动 6.9ms）**Phase 0 完成**
- **Phase 1 计算 tile**（图不变，cp=1 退化执行下对数值）：1.1 logits+topk 融合（**FP4 MXFP4 打分**，mxf4 UMMA + 局部 top-K 候选堆，logits 不落 HBM）→ 1.1b topk K 对齐 V4（**已验收**：`top_k` 编译期参数 512/1024，堆/overflow/scratch 容量 1:1 右尺寸化（q_len=2 省 ~40KB smem），Modal 双 K 全过：values OK + 索引重合 1.0000，merge 路径真实触发 K=512→3 次/chunk、K=1024→2 次/chunk）→ 1.2 attention（**已验收**：fork flashinfer `hca_fp8.py` 完成，`mega_dsa_cp/tiles/hca.py`——砍 persistent/split-KV/SM103/reduction/host 侧，grid (2, q_len, B) 一 CTA-pair 一 (b,t)，k_tile 数从 K_valid 标量推导，CSA cmp 流 page=1 gather（entry id 列表即页表、8 个一批发射），win/cmp 双流 + 7 pipeline + 双 softmax 组全保留；**按 S-bucket 全静态编译**（gotcha #33：动态 shape marshal 是 cutlass#2794 已知 bug；Q 侧固定、KV 侧 S 标量动态由 bucket 容量池 + k_valid/win_valid 设备标量承担）；torch 在线 softmax 模拟参照下 5/5 过：C128A 变长/attn_sink、CSA gather 512、partials 路径、LSE log2 域，详见 phase1.2-attention-tile.md）→ 1.3 compressor（含 FP4 量化写 index-K；**CP-native，无 cp=1 形态**——token 交错分片下压缩组跨 rank，partial stats + cmp_stats 推送 + finalize merge 的分解即算子本身，对应 `cmp_stats_arrived` 事件；Modal cp=2 验收，详见 phase1.3-compressor-tile.md）→ 1.4 通信任务实现 → 1.5 GEMM tiles
- **Phase 2 跨卡执行**：cp=2→4→8，数值双参照（自身 cp=1 + vLLM TP），性能实验矩阵
- **Phase 3（可选）**：DSL 化（tirx 方法论：队列字节等价+数值等价）、动态 MPMC 调度、2D 分解

## 运行环境（Modal）

本机只做编辑与 CPU 级校验；GPU 执行走 Modal（模式与 svdquant-kernels 一致：源码 `add_local_dir(copy=False)` 挂载、设备端 JIT，无本地构建）。入口 `scripts/modal_app.py`：`modal run scripts/modal_app.py::{smoke,single_gpu,dual_gpu,phase02_single,phase02_dual,phase03_dual,phase04_dual}`，日志 `log/<task>.log`。镜像 = debian_slim py3.12 + torch 2.11 cu130 + nvidia-cutlass-dsl + cuda-python。注意：Modal 屏蔽计数器级 profiling（ncu、`nsys --gpu-metrics-device` 不可用；torch.profiler / nsys 时间线可用）；每条命令设硬超时（kernel 秒级、编译 ~1 分钟，超时即死锁）；`retries=0` 防整套件重跑烧钱。双卡测试用 `gpu="B200:2"`（单节点 NVLink P2P + torch symm_mem rendezvous over nccl group）。

## 参考代码（tmp/）

文档：`docs/overview.md`（本文）、`docs/event-system.md`（Phase 0.1 spec）、`docs/gotchas.md`（已知坑与设计约束）、`docs/phase1-logits-tile.md`（Phase 1.1 FP4 logits+topK tile 设计）

- `tmp/flashinfer`（分支 pr-3943）：HCA kernel，`flashinfer/cute_dsl/attention/dsa/`
- `tmp/flashinfer-pr3381`：DSA compress+norm+rope CuTe DSL kernel
- `tmp/flashinfer/moe_ep/kernel_src/cutedsl_megamoe/`：多卡 megakernel 范式
- `tmp/tirx-kernels`（分支 agent/megakernel-dsl）：megakernel DSL、etensor、调度器
- `tmp/vllm`：mainline vLLM（V3.2 DCP、V4 模型）
- `tmp/vllm-pr44573`：V4 DCP 基线实现
- `tmp/vllm-ascend`：DSA-CP/SFA DCP/fused indexer
- `tmp/sglang`、`tmp/cutlass`

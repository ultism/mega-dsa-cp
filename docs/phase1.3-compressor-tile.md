# Phase 1.3: compressor tile（CP-native，无 cp=1 形态）

状态：**已验收**（2026-08-08，Modal B200:2 cp=2，300 步滚动全绿）。语义三方确认：sglang `c4_v2.cuh` / `c128_online_v2.cuh` / `fused_norm_rope_v2.cuh` + TRT-LLM blog26 + overview 冻结决策。

## 1. 模型语义规约（单机版，CP 分解前的基准语义）

### CSA（ratio=4，双流）

- **每 token 记录**（`wkv_gate` 投影产出，fp32）：`[kv_ov | kv_nm | s_ov | s_nm]` = 4×head_dim——**双投影分支**（blog26: "Consecutive outputs overlap by m raw-token positions through separate projection branches"）。token 在上一组当 overlap 成员（role 0）、本组当 normal 成员（role 1），两种角色各带 (kv, score)。
- **压缩窗** = 8 槽：overlap 组 4 槽（role 0 记录）+ 本组 4 槽（role 1 记录），2m=8 token 感受野，相邻窗重叠 4。
- **pooling**（`c4_forward`）：**per-channel** softmax——对通道 ch：w_j = exp(s_j[ch] + ape[j][ch] − max_j) ，out[ch] = Σ_j w_j·kv_j[ch] / Σ_j w_j。`ape` = (8, head_dim) 可学习表，按窗内槽位加偏置。
- **首组无 overlap**：seq_len==4 时 slots 0-3 无效（score=−inf 掩码，kv=0）。
- **触发**：decode 每追加 token，`seq_len % 4 == 0` 时本组关闭产出 1 条 entry。
- **attention 流**：head_dim=512（448 nope + 64 pre-rope 内嵌）→ 后处理 → fp8 e4m3 写压缩池（512B/entry，我们的 HCA 池无 per-entry scale，scale 折入 softmax_scale 外侧）。
- **indexer 流**：head_dim=128 → 后处理 → MXFP4 68B/entry（64B E2M1 + 4B UE8M0×4）写 index-K 池。

### C128A（ratio=128，online 单流，无 indexer）

- **每 token 记录**：`[kv | score]` = 2×512 fp32。**状态**：`[max | sum | kv_normalized]` = 3×512 fp32/序列。
- 每步：`pos = (seq_len−1) % 128`；per-channel online softmax 合并（max rescale；状态存**归一化后**的 kv + max/sum 供还原）；bias = ape128[pos][ch]（128×512 表）。
- `pos==127` chunk 关闭 → 输出状态 kv（已归一化）→ 后处理（同 attention 流）→ 写池；状态不更新（下一 chunk 首 token 重置）。
- `pos==0`：状态 = (score+bias, 1.0, kv)。

### 后处理链（两流共用，finalize 内融合）

RMS norm（全 head_dim，learnable weight）→ rope（**末 64 维**，position = 组首 `4n` / `128m`）→ **[indexer 流专用] Hadamard**（128 点，1/√128，warp butterfly）→ 量化写池。

- **Hadamard 决策：纳入**。正交变换保点积（q·k = (Hq)·(Hk)），checkpoint 无关；q 侧配套变换属 1.5 qpath（indexer q 的 norm/rope 后同样 Hadamard）。sglang 产品化路径同款（`fused_norm_rope_indexer_fp4`）。
- **scale clamp 不抄 sglang**：sglang 是 `fmax(1e-4, amax)/6`，我们冻结 DeepGEMM bit-identical `max(amax/6, 1e-4)` + UE8M0 ceil + 过 bf16 舍入（amax<6e-4 时两者差 6×，仅影响近零通道）。
- **rope 位置 = 组首**（sglang: `position = seq_len − compress_ratio`）。

## 2. CP 分解（partial/finalize 即算子本身）

token 交错分片（token t → rank t%cp）下，CSA 8 槽窗（cp=2：每 rank 4 槽）与 C128A 128-token chunk（每 rank 64）必然跨 rank。**没有 cp=1 形态**——分解即算子。

### CSA

- **partial tile**（每 rank 本地，每序列每步）：若本步本序列有组关闭（`seq_len % 4 == 0`，设备标量推导），对本地槽子集（cp=2 时 4 槽：2 overlap-role + 2 normal-role，槽位 ape 按窗内位置静态确定）计算 per-channel 未归一化 stats：`m_r[ch] = max(s+b)`、`w_r[ch] = Σ exp(s+b−m_r)·kv`、`l_r[ch] = Σ exp(s+b−m_r)`。
- **ring 缓冲**：每 rank/序列/流 存最近 4 条本地记录（CSA 窗本地恰 4 条；q_len 只影响"几条是新的"，ring 深度恒 4），fp32。
- **stats payload**：`valid + entry_id + 双流 × (m, l, w)` fp32 ≈ 7.7KB/组/rank。
- **推送**：每步每序列每 rank **无条件**推送（无组关闭则 valid=0）——事件 arity 静态 = cp（u64 相位计数器要求）；谓词在 payload 里。
- **属主映射**：**entry n → rank n % cp**（entry 交错；raw 坐标 4n%cp 恒 0 不可用作分片键）。属主 rank 的 HCA gather 本地 entry、LSE 合并兜底全局。
- **finalize tile**（属主 rank）：wait cp 份 stats → per-channel merge（m = max_r m_r；out = Σ_r e^{m_r−m}·w_r / Σ_r e^{m_r−m}·l_r）→ 后处理链 → 写本地压缩池 + index-K 池 → `K_valid[b] += 1`（设备标量）。

### C128A

- 各 rank 对 open chunk 纯本地累积状态（无通信）；chunk 关闭时**双方把状态推给属主**（entry m → rank m%cp；注意 closer ≠ owner 一半概率——token 128m+127 在 cp=2 恒落 rank 1）→ 属主 merge 双状态 → 后处理 → 写池。
- 状态重置：chunk 关闭后双方各自重置本地状态（时序确定，无通信）。

### 可见性时序

entry n 在 step t 关闭、finalize 完成 → **step t+1 起**对 logits/attn 可见（logits wait 上一步 finalize 事件；u64 单调相位计数器天然表达跨步依赖）。compressor 因此**不在当步关键路径**。

### 事件接线

`cmp_stats_arrived[rank]`（已冻结，sys scope，arity=cp）→ finalize tile。推送复用 Phase 0 通信原语（stats 批量拼 buffer + red.release flag）。流量：CSA 每步 ~B×cp×7.7KB ≈ 500KB（B=32, cp=2，全量无条件推送），与 logits candidates 同量级，可接受；C128A 摊薄可忽略。

## 3. 边界

- **输入**：kv_score 记录（GEMM/wkv_gate 产出，1.5 的事）；1.3 测试直接合成记录注入。
- **不做**：prefill、plan 机制（静态图 + 设备标量 seq_len 替代）、sglang 池布局（584B）、scale clamp 顺序。
- **SWA-only 层**：无 compressor（kvwrite 纯本地，已在骨架）。

## 4. 验收（Modal cp=2 双卡）

1. **CSA 双流**：多步滚动（≥3 组关闭），含首组无 overlap、变长 batch（S 错位使各序列不同步关闭）、q_len=2；对 torch 单点参考（完整 8 槽 pooling）数值逐值。
2. **C128A 状态机**：跨 chunk 边界（S=130±）滚动，关闭 + 重置 + 第二 chunk 累积正确。
3. **FP4 写**：对 `tiles/fp4.py` DeepGEMM oracle bit-identical；Hadamard 正交性 sanity（随机 x, y：(Hx)·(Hy) ≈ x·y）。
4. **stats 合并正确性**：同一随机输入，CP 分解结果 == 单机完整 pooling（fp32 容差内）。
5. **属主平衡**：entry 交错落两 rank，各 rank 池只含本地 entry。

产出：`mega_dsa_cp/tiles/compress.py`（partial + finalize + torch 参考）、`tests/test_compress.py`、Modal `phase13_dual` 入口。

## 5. 验收结果（`phase13_dual`，300 步 × B=8 × cp=2）

- micro bit-gates：staged Hadamard 与 MXFP4 量化对 torch 参考**逐位一致**
- CSA 双流合并 fp32：max err 1.2e-6；**index-K MXFP4 entry 端到端逐位一致**（0/38400 nibble 失配，sf 全对）
- attn fp8 entry：0 元素超 ±1 ulp 容差；C128A 合并 max err 2.1e-6；k_valid 计数器全对

## 6. 踩坑记录

1. **关闭检测必须是 step 级**（本步任一新 token 闭组则双 rank 都算+推 partial），不能是"我的 token 闭组"——否则合并永远缺一半 partial。C128A 同理（closer ≠ owner）。
2. **部分 warp 的 shuffle_sync 全掩码死锁**（gotcha #35）：`if tidx < 4:` 里 `shuffle_sync_bfly`（默认 mask 0xffffffff）要求 32 lane 全到——块内二级归约必须全 warp 参与 + 无效 lane 乘零。
3. 参考实现自身的 bug 会伪装成 kernel bug：C128A"kernel 错"实为 presim 状态机忘了 chunk 关闭后重置（kernel 是对的，与 CP 分解参考逐位吻合到 1e-7）。
4. DSL 事件通知的正确形态：notify 放顶层（EventSet 内部守卫单线程），与 test_comm 已验证模式一致。

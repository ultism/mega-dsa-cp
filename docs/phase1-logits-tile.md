# Phase 1.1 logits tile 设计（FP4 MXFP4 打分 + 融合局部 top-K）

状态：**v1（FP4 paged MQA logits）+ v2（融合精确 top-K 候选堆）均已在 B200 数值验收**（`tests/test_logits_fp4.py`，Modal `phase11_single`）：v1 logits vs torch fp32-dequant 参考 max rel 2.5e-7；v2 全部 8 组 (b,t,c) 候选值 OK + 索引集合 100% 重合。实现 `mega_dsa_cp/tiles/{fp4,logits}.py`。对应图中 `logits` 任务族（骨架里的 4 个 stub tile 的实体化与重新计数，见 §4）。

## 1. 契约

单个 logits task 处理**一个请求 b 的一段 rank-local KV chunk**，输出该段的精确局部 top-C 候选。

- 输入：
  - `q_fp4 / q_sf`：本 step indexer q，(heads=64 × q_len) 行 × 128 维，E2M1 + UE8M0(组32)，由 qprep 任务产出（wait 事件 `qprep_done`）
  - `weights[b]`：head-gate，heads×q_len 个 fp32（q_sf 不折叠进 weights，kernel 内应用——MXFP4 决策）
  - index-K cache：paged，page=64 entry，每 entry 68B（布局见 §2）
  - `block_table[b]`、`seq_len[b]`（设备侧标量，标量动态）
- 输出：候选 `(val fp32, idx u32)` × C（C = min(K, chunk_len)；K = 512（V4 Flash）/ 1024（V4 Pro），kernel 编译期参数），写入 arena cand buffer，notify `cand_ready[b, t, c]`（t = spec 槽位）
- 精确性：对段内全部分数精确 top-C（阈值流式选择，§5），段间由 cand 归并树精确合并（图既有结构）

## 2. FP4 数据布局

```
page（64 entry）= [64 × 64B E2M1 数据][64 × 4B UE8M0 scale]
entry i：数据 64B（128 维 × 4bit），scale 4B = 4 个 UE8M0（128/32 组）
page 字节数 4352B；TMA 粒度 = 2 page = 128 entry（8704B）
```

- `MmaMXF4Op` 约束（cutlass `tcgen05/mma.py:1766`）：A=B=E2M1、SF=UE8M0 vec32、K 指令=64、CtaGroup.ONE M=128、8≤N≤256 step8、**仅 K-major**、sm_100a/103a。KV entry 与 q 行都是 dim 连续 → K-major 天然满足
- UE8M0 scale 是纯指数：MMA 内 block-scale 直接消费，epilogue 不需要再做 dequant（对比 sglang FP8 路径的 per-token fp32 scale 乘——FP4 路径这个乘法消失，epilogue 更轻）
- compressor tile 写入时量化对齐 DeepGEMM：`scale = max(amax/6, 1e-4)` 的 UE8M0 ceil、过 bf16 舍入（保证与拆分路径 bit-identical）

## 3. MMA 配置与 TMEM 预算

- tiler：M=128（KV entry），K=128（=2×K64 指令），N = 64×q_len
- q_len=2（主配置 MTP-1）：N=128，单 MMA。acc 128 + SFA 8 + SFB 8 = **144/512 TMEM 列**
- q_len≤4：N≤256 单 MMA（acc 256 + SF 16+8 = 280 列）
- q_len=6-8（DSpark）：N=384-512 超界，**两遍 spec 半批**（t=0..2/3 然后 t=3/4..7，每遍 N≤256），KV smem 驻留复用，第二遍只重跑 MMA+epilogue
- SFA/SFB 走 TMEM：`blockscaled_utils.make_{smem,tmem}_layout_sf{a,b}`，S2T 用 `tcgen05.Cp4x32x128bOp` + `make_s2t_copy`（svdquant `kernel_v2_fa4.py:1600` 模式）
- 工厂：`sm100_utils.make_blockscaled_trivial_tiled_mma(Float4E2M1FN, K, K, Float8E8M0FNU, 32, CtaGroup.ONE, (128, N))`

## 4. 任务分解（对骨架的重新计数）

带宽论证（每层 indexer 扫描量 = S×B×68B；S=128K, B=32, cp=1 → 278MB，是层内最大单一流量之一）：

- 需要 ~100+ CTA 才能喂饱 HBM；**并行度取自 (B × kv-chunk)**，不是骨架的固定 4
- coords = `(b, c)`：b ∈ [0, B)，c ∈ [0, ceil(S/(cp×chunk_len)))；chunk_len 取 8K-16K entry
  - cp=8, S=128K：shard 16K → C=1，任务数=B（32-128）✓ 每任务串行 1.1MB
  - cp=1, S=128K：C=8-16，任务数=B×C（256-2048）→ 按 SM 数分波，静态队列按 S_bucket 枚举
- chunk_len 是 graph-build 常量；B、seqlen 是设备侧标量（越界任务谓词跳过照发 notify，标量动态原则不变）
- **骨架修订点**：`logits` 任务族从 4 个 stub 变为 B×C 个实体任务；cand 归并树 arity = C（每 (b,t)）；CP 候选交换与之后结构不变。事件表随 arity 参数化，拓扑族不变

单任务工作串：对其 chunk 内 128-entry 块流水（TMA → MMA → epilogue 插入堆），串行 64-128 块。

## 5. 融合 top-K 候选堆（epilogue）

- smem 驻留堆：`K × (fp32 val, u32 idx)`（K=1024 时 8KB/spec 槽位），外加 overflow 区 K（容量与堆 1:1 右尺寸化，1.1b 从 2048 降至 1024，q_len=2 共省 ~40KB smem）；维护当前阈值 θ（堆满后 = 堆内最小值，即精确第 K 名）
- 每块 128 分数（每 spec 槽位）：epilogue 的 relu2_fma 归约（svdquant/sglang 同款 packed f32x2）产出分数后：
  1. **seqlen 掩码**：kv_pos ≥ seq_len[b] 丢弃（替代 sglang 下游 lengths 掩码，几乎免费）
  2. **SWA 强制包含**：kv_pos ∈ [seq_len−128, seq_len) 不打分（sglang 的 +inf 技巧）——附带收益：logits 与 kvwrite（新 token 写入）**解耦**，不需要 wait kvwrite
  3. θ 过滤 → 命中者 warp 聚合 append 到 overflow
  4. overflow 将满（≥K−128）→ 合并（堆+overflow 重选 top-K，更新 θ）；合并次数 ≈ chunk_len/K 量级/任务
- 收尾：堆内 K 个按值排序（或仅写出无序 + val，归并树不在乎顺序），写 arena cand buffer，notify
- K > chunk_len 时（短上下文退化）：全量写出 + 计数，归并树兼容（对应 sglang naive_topk 捷径）

## 6. warp 组织与流水

单 CTA 内（每 task）：

| warp | 角色 |
|---|---|
| 1 TMA warp | KV+SF 融合块 TMA（2 page/块，一条 barrier，`extra_tx_count` 合并 tx）；Q/SFB/weights 进入口时一次载入 |
| 1 UMMA warp | per 块：S2T SF → `set(SFA/SFB)` → 2×`cute.gemm`（K64×2）；q_len>4 时第二遍重发射 |
| 4-8 math warps | TMEM LDTM → relu2_fma 归约（heads 维）→ §5 插入；weights 寄存器缓存（N≤128 全驻留） |

- KV 流水：stage 数 = smem 预算内尽量深（8704B/块 × 16 stage ≈ 136KB，配 32KB 堆 + Q/W ≈ 180KB，单 CTA 占满 228KB smem）——深流水补单 CTA 串行带宽
- acc pipeline：UMMA → math 用 PipelineUmmaAsync（svdquant 同款，producer 预置 phase=1 坑见 gotchas）
- logits **全程不落 HBM**（相对 sglang 两 kernel 方案省 2×S×4B×rows 流量）

## 7. 事件接口

- wait：`qprep_done`（q_fp4/sf/weights 就绪，整批单事件）
- 隐式依赖：index-K cache 的**历史**内容无事件（上 step 已写，kernel 边界同步保证）；本 step 新 token 由 SWA +inf 解耦（§5）
- notify：`cand_ready[b, t, c]`，arity=1，cand 归并树叶子
- arena 区域：`cand_buf[b, t, c]`（K×8B = 4/8KB，相位×2 轮换）、`q_fp4/q_sf/w`（qprep 产出区）

## 8. 数值验证

1. **logits 对拍**（tile 级）：同输入对 DeepGEMM `fp8_fp4_paged_mqa_logits`，fp32 分数逐值容差（UE8M0 量化一致时容差 ~1e-3 相对）
2. **top-K 对拍**：候选集合 vs torch 参考（fp32 dequant ground truth 的精确 top-K）——集合一致率 100%（同分数精度下），边界分数允许 ±1 替换
3. **E2E 召回**：vs vLLM/sglang FP8 路径的 top-512/1024 集合召回率（FP4↔FP8 差异是预期的，记录数值，预期 ≥99%）

## 9. 开放问题（实现期定）

- θ 更新与 overflow 合并的实测频率/开销（决定 overflow 区大小与合并实现：smem radix vs 排序）
- q_len=6-8：两遍 N=256 vs 三遍 N=128（TMEM 重利用 vs MMA 效率）
- B 大时 (b,c) 任务数超 SM 的波次内 cand 归并启动时机（归并 wait arity=C 全部，还是分层提前）
- chunk_len 常量随 S_bucket 的取值表（8K/16K 两档实测选）

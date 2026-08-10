# Phase 1.4a: 通信任务与 megakernel 集成契约（设计冻结稿）

Phase 1.4 = 把骨架（Phase 0.4）的 stub 通信边逐条换成真 tile，并补上两个从未实现的
merge tile。不含 qpath/GEMM（1.5）、跨层流水与调优（Phase 2）。依据：
`scripts/cta_size_study.py`（尺寸/占用/调度代数）、1.1/1.2/1.3 已验收 tile、
`docs/event-system.md`、`docs/overview.md` 通信清单。

## 1. Launch / CTA 契约（冻结）

- megakernel 统一 `block=512 threads`、`cluster=(2,1,1)` launch，grid.x 偶数；
  grid = 每 rank SM 数（B200=148），**1 CTA/SM**（统一 smem = max tile smem =
  HCA 208KB，bump allocator 顶层一次分配）。
- **任务两粒度**：HCA = cluster 级任务（`CtaGroup.TWO`，`hca.py:351`），静态发牌
  把同一任务（同 coords）写进同一 cluster 两个相邻 queue 的同一逻辑位置，两个
  CTA 各跑 cluster rank 0/1；其余全部 = CTA 级任务，在 cluster 内独立运行，
  **禁碰 cluster barrier**（含 cluster 范围的 pipeline init/sync）。
- regs 上限 = 64K/512 = **128 regs/thread**（统一编译取 max-of-paths）：
  各 tile standalone 编译产物的 regs/smem 需 GPU probe（cuFuncGetAttribute）
  确认可共存；probe 列入 1.4e 验收（见 §7）。
- intra-CTA warp-group packing（小 tile 复用 512 线程空位）量化为 ~2× CSA
  吞吐收益（S=1M/B=32 时 143→73µs），但需 barrier 改造（compress 用 CTA-wide
  `cute.arch.barrier`，`compress.py:230`）——**Phase 2 选项，1.4 不做**，
  调度接口不得排除该可能。
- 小 batch 下全层型为关键路径 bound（CSA 链 ≈19.6µs，其中 2×sys hop 5.7µs）：
  wait 一律 flag-only（gotcha #6），能折进 epilogue 的 push 不单独成任务。

## 2. 任务体重构契约

- 每个 tile kernel 重构为 `@cute.jit` device 函数，由 megakernel op dispatch
  按 op code + coords 调用；coords（b, t, c, …）来自静态队列，不再来自 bidx。
- tile 内动态 shape 禁令不变（gotcha #33）：S 相关上界由 bucket 常量承担，
  有效长度读设备标量（k_valid/win_valid/seq_len，同 1.2/1.3）。
- notify 一律放 kernel 顶层（1.3 教训：嵌进 `if tidx==0` 动态区会丢 peer
  notify）；EventSet CTA 级自守卫；短任务禁调 `producer_tail`（gotcha #28）。
- push 三段式不变：smem 暂存 → `fence_view_async_shared`+barrier → 单线程
  `push_start/push_finish`（gotcha #3）→ 顶层 notify。
- op code 表扩展：`QPATH_STUB / KVWRITE / LOGITS / MERGE_L1 / PUSH_CAND /
  MERGE_L2 / ATTN_HCA / LSE_MERGE / CMP_STEP / CMP_FIN`。（1.5 替换 QPATH_STUB，
  增加 GEMM 系。）

## 3. 事件表定稿（每层，cp 对称声明，wait 各自 cell）

| event | scope | arity | producer → consumer |
|---|---|---|---|
| `q_ready` | gpu | 1 | qpath stub（1.4 host 供 q）→ attn |
| `kv_written[b,t]` | gpu | 1 | kvwrite → attn |
| `l1_wait[b,t]` | gpu | C | logits chunk ×C → L1 merge |
| `cand_arrived[b,t]` | sys | cp | L1 merge（multimem.st + 双 rank notify）→ L2 |
| `topk_done[b,t]` | gpu | 1 | L2 merge → attn |
| `partin[b,t]` | sys | cp | attn partials push（双 rank）→ lse_merge |
| `lse_done[b,t]` | gpu | 1 | lse_merge → out（1.5: W_UV） |
| `cmp_stats_arrived[b]` | sys | cp | cmp step（双 rank）→ cmp finalize（1.3 已冻结） |

- C = ceil(ceil((S/cp)/128)/bpc)，按 S-bucket 编译期常量（S-bucket 契约）；
  事件格数（B=8,q=2,C=16）≈113 格 ×128B ≈ 15KB，远低预算。
- SWA 层：仅 `q_ready/kv_written/attn`（无选择、无压缩）；C128A 层：无
  logits/merge 系事件，attn 页表 = 压缩池全量，等 `kv_written` + 压缩池可见性
  （`k_valid` 标量，finalize 自增，同 1.3）。
- logits 仍不等 kvwrite（SWA +inf 技巧，1.1 已冻结）；attn 必须等。

## 4. Arena 布局（`alloc_arena`，phases=2）

| region | 内容 | 传输 | 尺寸（B=8,q=2,cp=2） |
|---|---|---|---|
| `cand_l1` | L1 输出：cp slices × (B,q) × (K×8B + count 8B) | multimem.st allgather | ~128KB |
| `inbox` | attn partials：(B,q) × cp slots × (O 256KB + LSE 2KB) | bulk S2G push（骨架已验证 4×64KB chunk） | ~8.2MB |
| `cmp_stats` | CSA b×cp×7.7KB + C128 b×cp×6KB（1.3 已定义） | bulk S2G + red.release | ~0.2MB |

chunk 级候选（logits 输出，(B,q,C,K)）与 L2 输出 `sel`（(B,q,K) entry-id
页表）为 rank 本地中间量，放普通 gmem，不进 arena。通信量核算：cand allgather
= B×q×(K×8B)×cp-fanout 与 S 无关（B=32 时 256KB，K=1024 时 512KB，合清单）。

## 5. 分层 topk（L1/L2 merge tile，1.4b 新实现）

- **为什么两层**：朴素单级 merge 的 allgather 量 = B×q×C×K×8B ∝ S（S=1M 时
  32MB/step，爆清单）；分层后推送量与 S 无关（每 rank 只推本地 top-K）。
- **无损性**：chunk 级留 K=K_global 条 → 单 chunk 最多贡献 K 条进全局，无损；
  rank 级同理。各层 exact top-K，整体与全量 top-K 集合一致。
- **L1**（每 rank 本地，grid 任务 (b,t)）：读 C 份变长候选（v fp32, i u32,
  count），复用 1.1 radix-select 机制对拼接数组 M=C×K 做 top-K；
  tie-break = logit 降序、entry_id 升序；输出 multimem.st 到 `cand_l1` 自己
  slice + notify 双 rank `cand_arrived[b,t]`。
- **L2**（每 rank 本地，(b,t)）：读 `cand_l1` 全部 cp slices（M=cp×K），同算法
  取 top-K，输出 `sel[b,t]`（确定性：双 rank 输入一致 + 算法确定 → 逐位相同，
  无需输出交换）+ notify `topk_done[b,t]`。
- K∈{512,1024} 编译期参数（同 1.1b）。SWA/C128A 层型无此 tile。

## 6. 各边接线细则

1. **logits → L1**：logits 任务化（coords=(b,t,c)），cand 输出由 torch tensor
   改普通 gmem 布局不变，notify `l1_wait[b,t]`（arity=C）。
2. **L1 → L2**：见 §5（multimem.st + sys notify）。
3. **L2 → attn**：`sel[b,t]` 作 HCA 的 page=1 gather 页表（1.2 已验收形态），
   attn waits += `topk_done`。
4. **attn → lse_merge**（sink 契约：仅 rank 0 的 partial 施加 attn_sink）：CP 语义下 q 各 rank 全量复制（清单：q 不交换），
   每 rank 对本地 KV 分片跑全部 (b,t) 的部分注意力，**每个 rank 都需要全局
   归并后的 O**（供后续本地 GEMM）。HCA acc partials epilogue（`hca.py`
   fp32 直出路径）把 fp32 O + log2 LSE bulk push 到**所有 rank** 的
   `inbox[b,t]` 的自己 slice（cp-1 次 peer push + 本地一次写，按 head-chunk
   切 4×64KB 流水，骨架已验证形态），notify 双 rank `partin[b,t]`
   （arity=cp，含 self-notify，同 1.3 双 rank 模式）。
   这是清单里 ld_reduce 的 **fallback 路径（bulk push + 本地归并）**，1.4
   取它（1.3 模式已验收）；ld_reduce 优化留 Phase 2，届时需两段式：LSE
   标量先 multimem.st allgather（清单 32KB 项）→ 本地算全局 max/scale →
   各自预缩放 O 后 switch 加法才成立（switch 只有 add，做不了非线性归并，
   gotcha #2）。
5. **LSE merge tile**（1.4c 新实现）：log2 域归并，语义照上游
   `reduction_kernel`（`hca_fp8.py:1591-1652`）：g = max_i l_i，
   scale_i = exp2(l_i − g)，O = Σ O_i·scale_i；输入 `inbox` cp 份
   （fp32 O + log2 LSE），输出 O_out（b,t,128,512）fp32 + 归并 LSE 到普通
   gmem，notify `lse_done[b,t]`。flag-only wait + scoped acquire 读（1.3 形态）。
6. **compressor 任务化**：step/finalize 四 kernel 改 op code（coords=b），
   事件/推送逻辑照搬 1.3 已验收实现（`cmp_stats_arrived[b]` arity=cp）。
7. **kvwrite**：host 供给的新 token KV/idx 记录写入 win 池/压缩池 + 更新
   `k_valid/win_valid` 标量 + notify `kv_written[b,t]`（1.5 后数据源换成
   wkv_gate GEMM epilogue，任务契约不变）。
8. **qpath stub**：host 供 q，`q_ready` 一发俱全（1.5 填真身）。

## 7. 验收计划

- **1.4b**：merge tile 单卡验收 —— L1/L2 同 kernel 两配置，vs torch.topk
  参考：集合重合 1.0000；确定性（同输入两跑逐位同）；变长 count（θ 过滤后
  <K）；K∈{512,1024}；M 覆盖 C×K 全范围（C=1..S_bucket 上限）。
- **1.4b 已验收（2026-08-08，Modal B200）**：`tiles/merge.py` 单 kernel 双配置
  （L1/L2 同构），256 线程/CTA（统一了设计表的 256/128 估计行）。实现事实：
  64-bit key = (sortable(logit)<<32)|~entry_id，顶位翻转进 signed 域；
  **boundary scan 顺序按 pass 区分**——pass 0 顶字节 signed 降序（127..0 接
  255..128），pass 1-7 普通 unsigned 降序（255..0），翻转只影响顶字节；
  early exit（boundary bin 恰好填满即停，实测 1-2 pass 收敛）；确定性
  two-pass strip emit（输入序输出，禁 atomic 乱序）。验收 10 例全 PASS：
  K∈{512,1024}、L∈{1,2,4,8,256}（L=256 即 S=1M/cp=2 的 chunk 数）、
  full/变长/tiny(M<K)/密集 tie/全等/-inf 混合，**有序逐位一致**（同集合同
  顺序）+ 双跑确定性。踩坑：m_total 归约曾踩 #32 变体（constexpr 循环内动态
  if 改循环携带变量）；扫描顺序 bug 见上。
- **1.4c 已验收（2026-08-08，Modal B200）**：`tiles/lse_merge.py`（256 线程，
  语义照上游 reduction kernel：scale_i = exp2(l_i − glse)，对**归一化**
  partials 精确成立）。6 例全 PASS：cp∈{2,4}（含 cp_max=8 填充路径）、
  sparse（35% 随机 -inf）、全空头（O=0、glse=-inf 边界正确）、B=32。
  实测 dO ≤ 5.3e-6、dLSE ≤ 9.6e-7——**容差从文档的 1e-6 修订为 1e-5**：
  fastmath exp2/log2 的 ~2^-22 相对噪声 × O 值域（~±8）即此量级，远小于
  下游 bf16/fp8 量化噪声，不值得为此换 precise 超越函数。
  **1.4d 接线注意**：attn_sink 只能由一个 rank 施加（rank 0，token 交错下
  position 0 属主），否则 sink 质量双计——写进 §6.4 契约。
- **1.4d**：骨架逐链接线 cp=2（沿用 Phase 0.4 harness + `codegen_schedule`
  扩 cluster 任务发牌）：先 topk 链（logits→L1→push→L2→sel 正确性），再
  attn-LSE 链（HCA partials→inbox→归并 vs 1.2 torch ref），再 cmp 链。
- **1.4e**：单层 E2E cp=2 × 300 步 × B=8（q=2），host 注入随机数据滚动
  验收：全链 torch 组合参考（1.1+1.2+1.3 ref 拼装），选择集重合 1.0000、
  O 输出 1e-5、压缩池逐位（同 1.3 门槛）；**GPU probe**：cuFuncGetAttribute
  测 megakernel 本体 regs ≤128/thread、smem ≤227KB，及各 standalone tile
  regs/smem 归档进本文档。
- Modal 入口：`phase14_merge` / `phase14_lse` / `phase14_chain` / `phase14_dual`。

## 8. 不做

qpath 与全部 GEMM（1.5）；跨层 prologue overlap（blog26 形态，Phase 2）；
intra-CTA packing（Phase 2）；61 层全流水与步进驱动（Phase 2）；staggered
start（队列中）；页粒度 ablation（exact baseline 之后）；cp>2 实测（Phase 2，
契约按 cp 参数化）。

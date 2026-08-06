# Phase 1.2 attention tile 设计（fork flashinfer hca_fp8.py）

来源：`tmp/flashinfer/flashinfer/cute_dsl/attention/dsa/hca_fp8.py`（3898 行，已全文精读）。选择语义见 overview.md「C4A 选择粒度决策」。

## 任务契约

任务坐标 `(b, t)`（b=batch 行，t=spec query 行，q_len≤8），一个任务 = 一个 CTA-pair（2-CTA cluster，CtaGroup.TWO）：

- **输入**
  - q_latent：qpath tile 产出，(H=128, 512, q_len, B) fp8 e4m3，每 query 一行
  - 双池 KV：`c_latent_win`（SWA 窗池，page=128 token）+ `c_latent_cmp`（压缩池，entry=512B fp8；**CSA 按 page=1 gather 消费、C128A 按池页枚举**），各带索引/页表；V = 同 latent 的 MN-major 转置视图（HCA 原样）
  - **cmp 索引列表**：CSA = topk-512/1024 **entry id 列表**（cand merge 树产出 (logit, entry_id)，entry_id 即压缩池逻辑槽位，直接当 1-entry 页的页表）；C128A = dense 位置推导页列表（ceil(S/128) 个 C128 条目，按压缩池 page 划分）；SWA-only（ratio 0）= 空列表退化
  - 有效长度：`window_valid_len`（≤128，由 seqlen 推导）+ `K_valid`（=128 + 选中 entry 数；CSA 因果由 logits 掩码保证——选中 entry 只含 ≤query 位置的 4-token 组；越界/填充 id=-1 由 K_valid 截断）
  - softmax_scale、output_scale、attn_sink（每 head 一个，V=0 虚拟 logit，HCA 原生支持）
- **输出**（接 CP LSE 合并任务）
  - O partial fp32 (H, 512, q_len, B) —— cp=1 时直接写最终 O（bf16）
  - LSE partial fp32 (H, q_len, B)，**log2 域**（HCA 原生：`log2(row_sum) + scale_log2·row_max`；CP 合并与 0.4 stub 同域）
- **同步**：wait = cand merge 完成事件 + qpath 完成事件 + kvwrite（win 池本步新 token）；notify = 本 rank LSE 合并（cp=1）或 partials push（cp>1）。standalone 阶段用普通 kernel 边界代替事件

## fork 范围

砍：
- persistent tile scheduler（`HCAStaticTileScheduler`）与 host 侧 `_compute_grid`/`get_split_kv`/`can_implement`/wrapper 校验
- split_kv 全部（`block_split_kvs`、workspace partials、reduction kernel 单独成 kernel）——我们的 CP 合并任务承担 reduction 语义
- SM103 分支（`LdRed32x32bOp` 硬件行归约），只留 SM100 路径
- var_seq/var_split 变体、非 causal 路径

留（核心资产）：
- 双流 trick：`k_index==0` 走 win 池、`k_index≥1` 走 cmp 池，两流共享 sKC/sVC smem、各持不同 page-tile 尺寸的 TMA view
- 16 warp 组织：softmax g0(0-3 偶 k-tile)/g1(12-15 奇)、correction(4-7)、mma_qk(8)、load_k(9)、load_v(10)、mma_pv(11)；寄存器 144/144/80
- 7 条 pipeline：load_q(1)/load_k(3)/load_v(2) PipelineTmaUmma；mma_s(2)/mma_o(2) PipelineUmmaAsync；p_mma(2) PipelineAsyncUmma；p_cor(2) PipelineAsync
- 双 softmax 组 TMEM 元数据交换（p_cor 4 列/stage：row_sum/row_max/correction_factor/no_correction）+ order_bar 定序 + skip_correction vote
- 掩码：槽位 <128 用 window_valid_len，否则 K_valid；attn_sink 注入
- TMEM 布局：S 2 stage(0-127) + O 256 列(128-383) + cor 元数据 8 列(384+)

改：
- **CSA cmp 流 gather 模式**：`page_size_cmp=1`（每 entry 512B fp8，满足 TMA 128B 对齐；原版 `can_implement` 拒 page_size≤1 是通用布局保守，fork 里放开）；`load_tma_qk_one_k_tile`/`load_tma_v_one_k_tile` 按 entry chunk 重构（k_idx 寄存器张量 128×4B 超 load warp 80 寄存器预算，改 8 个一批流水发射）；字节数与页版相同，损失空间局部性 + TMA 发射开销；若效率不达标，备用 FlashMLA sparse 式 LDGSTS gather
- k_tile 总数从 `cache_seqs` 推导改为：1（win）+ ceil(选中 entry 数/128)（CSA 固定 4/8 tile，C128A 标量动态 ceil(S/128)/…）
- cmp 索引来源：cand merge 输出 buffer（standalone 阶段由测试喂合成 entry id 列表）
- q 来源：qpath tile 输出 buffer（standalone 阶段合成）
- epilogue 输出：split_kv=1 路径（直接写 O/LSE）+ 新增 partials 路径（fp32 O+LSE 写 CP arena inbox，替代原 acc_o/acc_lse workspace——布局沿用 (H, split→rank, 512, q_len, B)）

## 与 CP 的映射

HCA 的 split-KV + reduction 与我们的 CP 同构：split partials ≡ rank partials，reduction kernel ≡ 跨卡 LSE 合并任务（全局 LSE = max + log2 Σ exp2(li−max)，scale_i = exp2(li−g)，O = Σ O_i·scale_i，reduction_kernel:1591-1652 照抄）。cp=1 时合并退化为直通（split_kv=1 路径原样）。

## 任务分解与骨架修订

skeleton 的 attn(s,h) 16 stub 修订为 **attn(b,t) = B×q_len 个实体任务**（同 1.1 的 logits 修订模式：坐标从 stub 变真实，拓扑族不变）。B=32, q_len=2 → 64 个 CTA-pair = 128 CTA，无需 head 切分（M=128 恰是 2-CTA MMA 的 M tile）也无需 split-KV（每任务 1 win + 4/8 个 cmp k-tile（Flash/Pro），C128A 按 S/128 标量动态）。

## 验收

1. standalone v1（cp=1 退化）：合成 q/KV/索引列表，C128A 变体（dense 全量，页枚举）对 torch 参考（fp32 dequant 双池 softmax）逐值
2. CSA 变体：合成 topk-512 entry id 列表（gather 模式），同参考（entry 级精确选择，与官方语义一致）
3. 掩码：window_valid_len <128、K_valid 截断、attn_sink、spec q_len=2 逐行因果
4. LSE log2 域校验；O/LSE partials 布局与 0.4 通信 stub 对齐
5. Modal `phase12_single` 入口；数值对拍记录进本文档

## 验收结果（2026-08-06，Modal B200，5/5 PASS）

standalone v1 全绿（`tests/test_hca.py`，`modal run scripts/modal_app.py::phase12_single`）：

| case | 内容 | O vs online-sim | LSE |
| --- | --- | --- | --- |
| tiny-c128a | 单 batch 单 tile + 死锁标记轮询 | 7.7e-1 ✓ | 3.9e-3 ✓ |
| c128a-nosink | B=4 q_len=2，S∈{4396,100,8229,2048} 变长（含 S<128 窗口残缺） | 8.0e-1 ✓ | 4.9e-3 ✓ |
| c128a-sink | 同上 + 有限 attn_sink | 8.0e-1 ✓ | 4.9e-3 ✓ |
| csa-gather | page_size=1 gather，topk-512 合成 entry 列表（5 k-tile） | 1.1 ✓ | 7.3e-3 ✓ |
| c128a-partials | acc_o/acc_lse partials 路径（0.4 stub 布局） | 1.1 ✓ | 7.8e-3 ✓ |

补充验证：C128A 4-tile（S=33K）、CSA 2-tile、CSA 5-tile（seed 变体）均过——gather 模式与 tile 数无系统性问题。

**验收方法关键**：torch 参考必须用**在线 softmax 模拟**（`ref_hca_online`：逐 tile running max + 每 tile fp8 P 量化 + exp2 校正 rescale）；naive final-max fp8P 参考有系统性单边差异（gotcha #34）。残余差是 ex2.approx 引发的 fp8 边界翻转，容差 rtol/atol 2e-2（|V|≤448 raw 下）。

**调试过程两个根因**（详见 gotcha #33）：
1. launch error 9（两天）= `grid.z` 用了 fake tensor 动态维度，marshal 解析为 0（NVIDIA/cutlass#2794）→ **按 S-bucket 全静态编译**：Q 侧固定（B、q_len），KV 侧 S 动态由 **bucket 容量**池/页表 + `k_valid`/`win_valid` 设备标量截断承担（拓扑静态、标量动态）；grid 不含 S；cache key 的 pages/cols 即 bucket 容量——**不是按精确 S 逐个编译**。调用契约：池与页表按 bucket 容量分配，装载只触达有效页（末 tile 越界槽位为池内陈旧有限值，掩码兜底）
2. XID 13 = 垃圾 stride 导致的非法 TMA 描述符，随静态化一并消失

**目前与设计的出入**：无功能性出入；V gather 的 dst 槽位按 `i_sub = slot // 64` 子块映射（pv tiler (128,256) 下每 pv_k 迭代 64 槽）。

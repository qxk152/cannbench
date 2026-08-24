# DeepSeek V3.2 DSA 三路径语义对齐说明

## 结论

CannBench 当前的 Ascend SIMT、vLLM-Ascend 和 CUDA 三条路径采用了相同的
DSA 两阶段抽象、逻辑输入输出和真实序列元数据合同。

- 三条路径都执行 `Lightning Indexer -> indices -> Sparse Attention`。
- 三条路径从同一套 padded canonical 输入和逐请求长度生成各自后端需要的
  物理布局；跨机器 artifact 和 conformance 门禁已经建立。
- Ascend SIMT 与 vLLM-Ascend 的完整 V3.2 prefill/decode 已通过门禁；CUDA
  路径尚缺可用 NVIDIA 节点，当前只有 adapter 与 runner 覆盖。
- V3.2 BF16 的 vLLM-Ascend 路径最终执行 CANN OPP 随包提供的预编译 device
  kernel；自编译的 vLLM Ascend 扩展在该路径中是 host 侧 wrapper。因此不再把
  直接 CANN API 调用作为一条独立的 V3.2 性能实现。
- 底层布局、存储精度和 kernel 数量可以不同，这些属于实现差异。
- 在 CUDA conformance 收敛前，性能测试只能作为初步数据，不能视为严格公平的
  三后端生产实现对标。

本文只讨论 DeepSeek V3.2。V4/V4 Pro、attention sink、SWA 以及完整 Attention
Layer 的其他前后处理不在当前对齐范围内。

## DSA 背景

普通 Attention 需要让每个 Query 与全部 `C` 个历史 Token 计算分数，计算量
近似为：

```text
O(B * Q * H * C * D)
```

DeepSeek Sparse Attention 使用一个较轻量的 Lightning Indexer，先从 `C` 个
KV 位置中选择 `TopK` 个候选位置，再只对这些位置执行正式 Attention：

```text
Lightning Indexer -> indices -> Sparse Attention
```

V3.2 当前保留的真实 case 使用以下主要参数：

```text
Lightning Indexer:
  index_heads G = 64
  index_dim   Di = 128
  top_k       S = 2048

Sparse Attention:
  query_heads H = 128
  qk_head_dim   = 576
  value_dim     = 512
  kv_heads      = 1 (shared-KV)

Decode:
  B=2, Q=2, C=32768

Prefill:
  B=1, Q=4096, C=32768
```

Indexer Query 与正式 Attention Query 不是同一个张量：

- `Q_index` 是 Indexer 使用的低成本索引 Query，逻辑维度为 `64 x 128`。
- `Q_attn` 是正式 Attention Query，逻辑维度为 `128 x 576`。
- 真实模型中二者来自同一层输入的不同投影。
- CannBench 当前分别生成 Indexer 和 Attention 输入，只将 `indices` 在设备侧串联。

因此，当前 workflow 测试的是 DSA 核心链路，而不是包含 projection、RoPE
生成、cache scatter 等操作的完整 Transformer Attention Layer。

## 目标逻辑接口

### Lightning Indexer

```text
输入:
  query   [B, Q, G, Di]  BF16
  keys    [B, C, Di]     BF16
  weights [B, Q, G]      BF16

计算:
  raw_score[b,q,c] =
      sum_g(weights[b,q,g] * relu(dot(query[b,q,g], keys[b,c])))
  score[b,q,c] = score_scale * raw_score[b,q,c]

输出:
  indices [B, Q, S]      INT32
```

V3.2 的 `score_scale` 固定为 `1.0`。`indices` 应按分数从高到低排列，并且
只能包含当前 Query 可见的 KV 位置。不同分数必须保持降序；同分数边界按
`equivalent_score_set` 比较，即任一同分候选集合均满足公开合同。SIMT 的
lowest-index 顺序只是确定性实现细节，不作为厂商原生 Top-K 的接口要求。

### Sparse Attention

```text
输入:
  query     [B, H, Q, 576] BF16
  shared_kv [B, 1, C, 576] BF16
  indices   [B, Q, S]      INT32

计算:
  keys = shared_kv
  values = shared_kv[..., :512]
  selected_kv = gather(keys, values, indices)
  scores = softmax_scale * (query @ selected_keys^T)
  probabilities = softmax(scores)
  out = probabilities @ selected_values

输出:
  out [B, Q, H, 512] BF16
  lse [B, Q, H]      FP32 (natural-log logsumexp)
```

V3.2 的 `softmax_scale` 固定为 `576^-0.5`。每个 `[B,Q]` 行通过
`topk_lengths` 声明有效候选数，长度外或超出 KV 范围的 index 使用 `-1`
语义处理。全无效行的 output 为 0，LSE 为 `-inf`。当前 V3.2 realistic case
均使用完整 2048 个候选；SIMT 融合 kernel 因此只接受满 Top-K 行和默认 scale。

Workflow 由两个独立算子组成，只把 Indexer 产生的 `indices` 直接绑定给
Sparse Attention。当前目标不是把整个 workflow 合并为一个 kernel。

因为这是两个 kernel，`indices` 必须作为 device tensor 保存在设备全局内存中；
要求是它不能回到 host，也不能由第二个算子重新随机生成或重新物化。

## 真实序列元数据

Canonical 张量使用 padded batch 表示，case 同时携带每条请求的真实长度：

```text
query_lens             [B]
context_lens           [B]
query_start_positions  [B]
cu_seqlens_q            [B+1]
cu_seqlens_kv           [B+1]
page_block_size         scalar
block_tables            [B, ceil(C_padded / page_block_size)]
```

其中 `query_start_positions[b]` 是该请求第一条 Query 的绝对 Token 位置；V3.2
使用 right-aligned causal 语义，因此默认值为
`context_lens[b] - query_lens[b]`。`cu_seqlens` 是真实长度的前缀和，例如
`query_lens=(1, 3)` 对应 `cu_seqlens_q=(0, 1, 4)`。它用于把 padded BQ 张量
无歧义地打包成 TND。

`block_tables[b, i]` 把请求内第 `i` 个逻辑 KV page 映射到物理 page；不足
`C_padded` 的尾部 entry 使用 `-1`。CannBench 生成的 canonical cache 当前采用
各请求独立、顺序分配的 page，adapter 不再自行假定所有请求具有相同真实长度。

Padding 行遵循固定公开语义：Indexer 返回 `-1` indices；Sparse Attention 的
`topk_lengths` 为 0、output 为 0、LSE 为 `-inf`。Workflow 构建时会校验两阶段
的真实 Query 长度、Context 长度、绝对位置、page 大小和 block table 完全一致。

各后端按以下方式 lowering：

- PyTorch baseline 保留 padded 张量，用逐行有效长度构造 mask。
- vLLM-Ascend Indexer 将有效 Query/KV 行打包成 TND；Sparse Attention 将有效
  Query 打包成 TND，并使用 canonical paged KV、block table 和逐请求 KV 长度。
- CUDA prefill 将有效 Query 行打包后调用 FlashMLA，并把 TND 输出回填成统一的
  `[B,Q,H,Dv]` / `[B,Q,H]`；paged decode 使用 canonical block table。
- SIMT 对 uniform batch 保持一次融合 kernel 调用；ragged batch 按请求切片调用
  同一个融合 kernel，并在 device 上组装 padded 输出，不经过 host。

## 三条实现路径

### Ascend SIMT

Lightning Indexer 是一个自定义 CV 融合 kernel：Cube 计算点积，结果从 L0C
搬运到 UB，SIMT VF 在 UB 上完成 ReLU、权重乘法、归约和 TopK。中间 logits
不写入 GM。

Sparse Attention 也是一个自定义 CV 融合 kernel：在一个 kernel 内完成选中
KV 的读取、Cube QK、UB 上的 Softmax/LSE，以及与 V 的乘加。两个组件之间只
通过 GM 中的最终 `indices` 连接。

该路径的公开逻辑输入为 BF16；具体的 BQHD/BHTD 布局由自定义算子 wrapper
约定。Materializer、wrapper 和自定义算子 ABI 均只接收单一 `shared_kv`，
kernel 直接用完整维度作为 K，并用每个 token 的前 `value_head_dim` 维作为 V。

### vLLM-Ascend

Lightning Indexer 调用 `torch_npu.npu_lightning_indexer`，使用 TND 布局、
累计序列长度和 `sparse_mode=3`。原生接口可能返回 indices 和 values，CannBench
当前只保留 indices。

V3.2 Sparse Attention 调用 `npu_sparse_flash_attention`。Adapter 将逻辑上的
576 维 Q/K 拆成 512 维 NoPE 部分和 64 维 RoPE 部分。当前统一 output/LSE
合同使用 TND KV 单次调用，因为 CANN 不支持 `PA_BSND` 在
`return_softmax_lse=True` 时返回 LSE。

这些 TND KV 和 NoPE/RoPE 拆分属于物理接口适配，不应改变逻辑算子语义。
生产推理只需要 output 时仍可使用分页 KV，但不能把“paged output 一次 + TND
LSE 一次”的重复计算计入统一单次性能结果。

#### V3.2 BF16 的 device kernel 来源

在 Ascend 950PR、CANN 9.2.0 和 vLLM Ascend commit `059611dc1` 上，对
`deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` 的采集确认：V3.2 BF16
路径的 API 入口虽然属于 vLLM Ascend，但最终执行的是 CANN OPP 随包提供的
官方预编译 device kernel：

```text
Python / vLLM API
  -> self-compiled vllm_ascend_C.so Torch wrapper
  -> ACLNN / system libopapi.so
  -> official CANN OPP precompiled device .o
```

当前 CannBench 的 Indexer adapter 已直接调用
`torch_npu.npu_lightning_indexer`。vLLM Ascend 自身的
`npu_lightning_indexer` 和 `npu_sparse_flash_attention` wrapper 分别通过
`EXEC_NPU_CMD(aclnnLightningIndexer, ...)` 和
`EXEC_NPU_CMD(aclnnSparseFlashAttention, ...)` 下发；`vllm_ascend_C.so` 运行时
解析到 `/usr/local/Ascend/cann-9.2.0/lib64/libopapi.so`。因此这里“自己编译”的
是 Torch 注册、参数适配和 ACLNN 调用 wrapper，不是这两个 BF16 device
kernel 本体。

采集命中的主 kernel 来自以下 CANN OPP 二进制：

```text
/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/lightning_indexer/LightningIndexer_2517851d7f53b28f971ff94bfa4b7037.o
/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/sparse_flash_attention/SparseFlashAttention_9ec0bacf213f2b2dce3ba70149a4903b.o
```

直接 CANN API 临时验证路径与 vLLM-Ascend 路径的 kernel 序列、kernel hash、
tiling key 和 launch geometry 均相同；两个主 kernel 都是 `block=32`、
`mix_block=64`。同机采集的耗时如下：

| 组件 | 直接 CANN API | vLLM-Ascend |
| --- | ---: | ---: |
| Lightning Indexer | 98.247 us | 98.026 us |
| Sparse Attention path | 71.179 us | 71.059 us |
| Workflow | 169.426 us | 169.085 us |

这组差异属于 profiler 波动，不能视为两个 device 实现的性能差异。因此单独发布
`cann_ops_library` V3.2 记录会重复测量同一组 device kernel，只保留
`vllm_ascend` 记录。该结论只适用于 V3.2 BF16 DSA；vLLM Ascend 仍包含其他
自定义算子，以及量化/V4 DSA 的自定义 device kernel，不能概括为“vLLM
Ascend 只有量化算子”。

### CUDA

Lightning Indexer 先将统一 BF16 输入量化为 DeepGEMM 所需的 FP8 表示，然后
由 DeepGEMM MQA logits kernel 计算分数，再调用 `torch.topk` 产生 indices。
这一路径至少包含两个 launch，并且 FP32 logits 会经过 GM，因此目前不是
score 与 TopK 的单 kernel 融合实现。

Sparse Attention 的 prefill 使用 `flash_mla_sparse_fwd`，decode 使用
`flash_mla_with_kvcache`。Decode adapter 会把 KV 转换为 FlashMLA V3.2 使用的
FP8 cache 格式。FlashMLA 假定传入的 indices 已经满足有效范围，不再为稀疏
候选补做完整的因果过滤。

## 逻辑差异与物理差异

以下物理差异是允许的：

- SIMT 使用 Ascend Cube、UB 和 SIMT VF。
- vLLM-Ascend 的统一 output/LSE 路径使用 TND 和 CANN 原生算子布局。
- CUDA Indexer 和 Decode KV 内部使用 FP8，FlashMLA 使用自己的 cache 布局。
- 三条路径的 kernel 数量和融合程度不同。

这些差异不应改变对外张量代表的数学对象。Adapter 应负责布局和存储格式的
转换。

三条路径的公开返回值已统一为：

```text
(out: BF16 [B,Q,H,512], lse: FP32 [B,Q,H])
```

FlashMLA prefill adapter 丢弃私有 `max_logits`；vLLM-Ascend adapter 从原生
`softmax_max + log(softmax_sum)` 规范化出自然对数 LSE。

### 精度与结果比较

虽然三条路径可以使用同一份 BF16 逻辑输入，CUDA 路径内部的 FP8 量化仍可能
改变临界位置的 TopK 排序。因此验证不能只要求 indices 逐项完全相同：

- 单独验证 Indexer 时，应比较 TopK recall、重叠率和选中分数误差。
- 单独验证 Sparse Attention 时，应给三条路径传入同一份合法 indices。
- 验证完整 workflow 时，应比较最终 output/LSE，并记录 Indexer recall。

## V3.2 最终契约

当前建议将 V3.2 固定为以下契约：

```text
Lightning Indexer:
  BF16 canonical inputs
  right-aligned causal valid length
  INT32 [B,Q,TopK] sorted indices

Sparse Attention:
  BF16 canonical query/shared_kv
  K = shared_kv
  V = shared_kv[..., :512]
  INT32 indices
  softmax_scale = 576^-0.5
  invalid index = -1
  normalized (BF16 out [B,Q,H,Dv], FP32 natural-log lse [B,Q,H])

Workflow:
  indices 在设备侧从 Indexer 传给 Sparse Attention
  不包含 attention sink
  不包含 projection、RoPE 生成、cache scatter 和 SWA
```

三条路径现在具有相同的逻辑输入、输出和序列元数据合同。进入严格性能对标前，
仍需完成 CUDA 实机 conformance。

两条 V3.2 case 均来自 FlashMLA 单卡算子测试。现有 `batch`、Indexer
`index_heads`、Attention `query_heads/kv_heads` 和 `context_tokens` 都是
rank-local 维度；case 显式记录：

```text
shape_scope = rank_local
TP = 1, DP = 1, CP = 1
kv_shard = replicated
```

Lightning Indexer 与 Sparse Attention 的 workflow 配对会校验上述字段完全一致。
这保证三后端处理相同的单卡数据量，但不声称代表未知多卡部署的 local shape。

## 完整精度与 Conformance 状态

Conformance runner 使用 `splitmix64-period-v2` 生成器，通过 case、seed 和固定
周期在不同机器上重建同一份 canonical BF16 输入。Indexer 保存完整 INT32
indices；Attention 和 workflow 对 decode 验证全部 Query，对 prefill 验证
`0/1365/2730/4095` 四个 Query，并覆盖全部 128 个 Attention Head。

旧的线性周期生成器会让不同 seed 只产生同一序列的循环平移，造成 BF16 Top-K
边界出现数千个同分候选。它会把不同但合法的 tie 选择误报为低 recall，因此不再
用于 conformance。

SIMT 自身的完整 `576/512` 精度结果如下：

| Phase | Case | Output mismatch | LSE mismatch |
| --- | --- | ---: | ---: |
| decode | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` | 0 / 262144 | 0 / 512 |
| prefill | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` | 0 / 262144 | 0 / 512 |

SIMT 与 vLLM-Ascend 使用同 input 的门禁结果如下。Indexer 门槛要求每行及整体
Top-K recall 都不低于 `0.95`；数值输出使用 `atol=0.05, rtol=0.05`。

| Phase | Indexer mean recall | Indexer min recall | Attention output/LSE mismatch | Workflow output/LSE mismatch | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| decode | 0.986938 | 0.979980 | 0 / 0 | 0 / 0 | 通过 |
| prefill | 0.989263 | 0.973145 | 0 / 0 | 0 / 0 | 通过 |

CUDA runner 已连接现有 DeepGEMM Indexer、`torch.topk` 和 FlashMLA adapter，
但当前控制机没有 NVIDIA runtime，也没有可用 NVIDIA remote endpoint。因此
CUDA artifact 尚未生成，不能宣称三后端实机 conformance 已通过。

## V3.2 剩余工作

本节只记录尚未闭环的事项，不保留已完成事项或历史提交清单。

### 1. 完成 CUDA 实机 Conformance

当前没有可用 NVIDIA 节点，本项处于环境阻塞状态。获得 GPU 环境后执行以下实验，
不再补充其他 V3.2 功能项。

#### 实验环境

- 使用支持当前 DeepGEMM 和 FlashMLA kernel 的 NVIDIA GPU，记录 GPU 型号、
  CUDA、Driver、PyTorch、DeepGEMM、FlashMLA 和 CannBench 的精确版本或 commit。
- 安装 DeepGEMM、FlashMLA 及 CannBench CUDA adapter，先运行 adapter 单测和
  两个组件的最小 runtime smoke。
- 使用与 Ascend artifact 相同的 case、seed 和 `splitmix64-period-v2` canonical
  BF16 输入生成规则；禁止单独为 CUDA 重新随机生成输入。

#### 必跑 Case

| Phase | Case |
| --- | --- |
| decode | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` |
| prefill | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` |

两个 case 都必须分别生成 CUDA Indexer、Sparse Attention 和完整 workflow
artifact。单独验证 Attention 时使用与 Ascend 相同的合法 indices；验证 workflow
时使用 CUDA Indexer 的设备侧 Top-K 输出，不能经 Host 重新物化 indices。

#### 精度门禁

- Indexer：逐行及整体 Top-K recall 均不低于 `0.95`，并保存 mean/min recall。
- Sparse Attention：统一还原为 BF16 `[B,Q,H,512]` output 和 FP32
  `[B,Q,H]` natural-log LSE，使用 `atol=0.05, rtol=0.05` 比较。
- Workflow：比较最终 output/LSE，同时记录由 CUDA FP8 Indexer 引入的 Top-K
  recall；两个 phase 都必须通过。

#### Profile 审计

- 使用 NCU 采集，不使用 Host wall time 作为正式 latency。
- 核对 plugin 固定筛选规则覆盖 Q BF16-to-FP8 cast、DeepGEMM logits、
  `torch.topk`、动态 cache-index lowering 和 FlashMLA Sparse Attention 的实际
  kernel 名；发现未命中 kernel 时先修正规则再产出性能结果。
- 确认静态 Index-KV/shared-KV packing、FP8 cache 和固定 metadata 位于计时外。
- 分别保存 Indexer latency、Sparse Attention latency 和二者之和的 workflow
  latency，并保留原始 NCU report、解析后的 profile summary 和 conformance
  artifact。

#### 完成标准

只有 decode 与 prefill 的三层精度门禁全部通过、NCU 动态 kernel 列表审计完成，
并且原始与汇总 artifact 均已保存，才可删除本节并宣称 V3.2 三后端 conformance
和性能计时边界完全闭环。

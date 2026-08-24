# Softmax SIMT V3 优化思路整理

本文以 Softmax SIMT V3 目录的提交记录为主线，结合 PyTorch CUDA
Softmax 的 shape 分发路径、仓库内 V1/V2 说明、编译器最小复现和已发布
benchmark 数据，整理 V3 已完成的优化以及后续可继续验证的方向。

文中的状态分为两类：

- **已实现**：当前 V3 源码中已经存在，并标注引入或修正它的 commit。
- **后续建议**：基于当前实现和数据提出，尚未在 V3 中完成。

## 阅读约定与术语

| 术语 | 本文含义 |
| --- | --- |
| GM | Global Memory，设备全局内存。 |
| UB | Unified Buffer，计算 core 本地的片上缓冲区。 |
| SIMD / SIMT | 分别指单指令多数据和单指令多线程执行；V3 由 vector 外层 kernel 调用 SIMT vector function。 |
| VF | 由 `asc_vf_call` 启动的 SIMT vector function。 |
| MTE2 / MTE3 | 分别负责 GM -> UB 与 UB -> GM 的搬运流水。 |
| ILP | Instruction-Level Parallelism；本文主要指每个线程一次循环展开处理的元素数。 |
| persistent path | 让一行的输入或中间值尽量驻留在线程寄存器/片上存储中的 row-wise 路径。 |
| spatial path | 归约轴之后仍有非平凡 inner tail，即 `inner_size != 1` 时使用的路径。 |
| occupancy | 受线程数、寄存器和 UB 等资源约束，可同时驻留并执行的 block 程度。 |
| ASC 翻译单元 | 单个 `.asc` 源文件独立编译形成的 object；同一文件中的模板实例会共同影响编译器资源估算。 |

本文所称“logits 路径”只是“常见于模型 logits 的大 row shape”的简称。
实现不会识别 case 名或 logits 语义，只根据 dtype、shape 和 workspace 条件
分发。

## 1. 结论摘要

V3 的优化主线不是简单地给单个 kernel 调参，而是逐步建立适合不同
row shape 的多路径实现：

1. 继承 V2 的 CUDA 风格分发，把 row-wise Softmax 分成 persistent 和
   fast 两类路径，以 spatial 路径处理 `inner_size != 1` 的情况。
2. 把 Ascend 上的物理 grid 数量和每个 block 的线程数从固定策略改成
   shape-aware 策略，避免把 CUDA 的 launch 参数机械照搬到 Ascend。
3. 对 fast row 路径增加 ILP、`half2` 向量访问和寄存器缓存，减少全局
   内存访问和循环控制开销。
4. 针对高频的 `dim_size=512/1024`，先隔离编译单元规避编译器 UB
   估算问题，再引入 GM -> UB -> SIMT 计算 -> GM 的双缓冲流水。
5. 针对 logits 大 row，按“一整行能否放入 UB”继续分流：能放入时使用
   whole-row UB recompute；放不下时，FP16 使用单 kernel 融合 tiled stats/write
   并复用一块完整 UB tile，在线统计量保留在 UB；FP32 保留两个 MTE/UB tiled
   kernel fallback，并使用行级 GM workspace 连接两个阶段。

这条路径与 PyTorch CUDA 的经验一致：Softmax 的关键不是找到一个覆盖所有
shape 的万能 kernel，而是围绕 row 长度、对齐、片上存储容量和寄存器压力
建立可解释的分发边界。

## 2. PyTorch CUDA 提供的优化参照

仓库现有的 `../README.zh-CN.md` 总结了 PyTorch CUDA Softmax 的主要
前向分发。与 V3 最相关的路径如下：

```text
inner_size == 1:
    小/中等 row，且 row bytes 较小
        -> persistent warp softmax

    大 row
        -> fast global-memory / unaligned fast path
        -> register path
        -> shared-memory path
        -> generic row fallback

inner_size != 1:
    -> spatial softmax
```

可以从中提炼出四条可迁移到 Ascend 的方法论：

- **先按 shape 分发，再优化 kernel**：row-wise 与 spatial 的并行结构不同。
- **让中间值驻留在离计算单元更近的位置**：优先寄存器，其次 UB，最后 GM。
- **用专用变体换取编译期展开**：常见 row length 值得独立实例化。
- **分发阈值必须服从目标硬件**：CUDA 的 warp、block 和 shared memory
  参数只能作为算法参照，不能直接视为 Ascend 的最优 launch policy。

## 3. V3 当前分发总览

对连续输入，V3 根据归约轴将 tensor 展平为：

```text
outer_size x dim_size x inner_size
```

其中 `outer_size` 是归约轴之前各维度的乘积，`dim_size` 是归约轴长度，
`inner_size` 是归约轴之后各维度的乘积。后文所有 persistent/fast 示例都以
`inner_size == 1` 为前提；只要 `inner_size != 1`，无论 `dim_size` 是多少
都走 spatial path。

当前代码中的主分发可以概括为：

```text
if inner_size != 1:
    spatial path
else:
    if dim_size <= 2048 and dim_size * sizeof(dtype) <= 8192:
        persistent path
        if dim_size < 32:
            通用 persistent register/shuffle kernel
        elif dim_size <= 128:
            64/128 bucket padded-UB kernel
        elif dim_size <= 256:
            256 bucket padded-UB kernel
        elif dim_size <= 512:
            512 bucket padded-UB kernel
        elif dim_size == 1024:
            1024 专用 GM/UB 双缓冲 kernel
        else:
            通用 persistent register/shuffle kernel
    else:
        fast row path，block_x = 1024
        if fp16 and 8192 <= dim_size <= 56320:
            whole-row UB recompute path
        elif fp32 and 8192 <= dim_size <= 28160:
            whole-row UB recompute path
        elif row 超过对应 whole-row UB 上限:
            if fp16:
                单 kernel MTE/UB tiled stats/write + 完整 tile 复用
            else:
                两 kernel MTE/UB tiled stats/write fallback
        elif 2 <= ceil(dim_size / 1024) <= 8:
            register-cache path
        else:
            direct ILP + half2/scalar fast kernel
```

判断顺序也是优先级：whole-row UB 和 tiled workspace 高于 register-cache。
在后续两个大-row 分支加入后，当前受支持的 fp16/fp32 shape 会先被
persistent、register-cache、whole-row UB 或 tiled workspace 覆盖，因此最后的
direct fast kernel 不再由主分发命中。它的 ILP、x4/x2 和 scalar 访问助手
作为通用 fallback 实现继续保留。

代码还保留了 `GenericLike` 枚举和防御性报错，但当前 fp16/fp32 selector
不会返回该枚举；这里的 fail-fast 用于防止未来扩展 selector 后静默进入
未经准确率验证的实现。

以 `inner_size == 1`、fp16 为例：

| `dim_size` | 当前路径 |
| ---: | --- |
| `9` | 通用 1024-thread persistent register/shuffle |
| `48` | `persistent_128` 中的 64 bucket padded-UB |
| `128` | `persistent_128` 中的 128 bucket padded-UB |
| `196/204/256` | `persistent_256` 中的 256 bucket padded-UB |
| `512` | `persistent_512` 中的 512 bucket padded-UB |
| `1024` | `persistent_1024` 专用 GM/UB 双缓冲 |
| `32128` | fast row 下的 whole-row UB recompute |
| `50265` | fast row 下的 whole-row UB 双缓冲流水 |
| `128112` | fast row 下的 FP16 单 kernel tiled stats/write + 完整 tile 复用 |

## 4. 按提交记录梳理优化演进

### 4.1 建立独立 V3 基线（`0c24a09`）

**已实现。** V3 最初由 V2 复制并更换 namespace、注册名和包名得到，核心
kernel 与 V2 基本相同。这个提交的重要意义是建立独立实验边界，使后续
launch policy、编译单元和混合流水的调整不会改变 V2。

V3 基线已经具备 V2 的三项能力：

- `inner_size == 1` 与 spatial 路径分离；
- `dim_size <= 2048` 且 row bytes 不超过 `8192` 时走 persistent；
- 大 row 走 fast path，并保留 fp16 `half2` x4 访问。

因此，`0c24a09` 是版本隔离提交，不应被解读为一次性能提升。

### 4.2 提高 persistent/fast 并行规模（`6c7d930`）

**已实现。** 该提交把 persistent 的目标线程数从 `128` 提高到 `1024`，
把 fast path 从 `512` 线程提高到 `1024` 线程，并给 persistent kernel
增加 `__launch_bounds__(1024)`。

优化意图是增加单个 block 内可同时处理的 row 数和大 row 的归约并行度。
它也暴露出后续必须解决的问题：所有 shape 固定使用 `1024` 线程会带来
寄存器、UB 和编译器资源估算压力，不能作为最终策略。

### 4.3 切换到混合 SIMD/SIMT launch，并限制物理 grid（`8bffa96`）

**已实现。** 该提交把计算函数改为 `__simt_vf__`，由
`__global__ __vector__` 外层 kernel 通过 `asc_vf_call` 启动。它同时完成：

- 显式标注 GM/UB 地址空间；
- 将 spatial 二维 grid 展平为物理一维 grid，再在 kernel 内恢复
  `block_index_x/block_index_y`；
- 对 `dim_size <= 256` 的 persistent path，把物理 `grid.x` 限制为 `64`；
- fast path 继续按 `outer_size` 分块，并保留 `32768` 的 grid 上限。

这里的关键优化不是语法替换，而是把“逻辑工作量”和“物理启动规模”分开。
kernel 内通过 grid-stride loop 覆盖全部 row，物理 block 数则按硬件并发能力
控制，避免大量短 block 带来的 launch 和调度成本。

### 4.4 按 row length 调整 persistent block（`7125e85`）

**已实现。** 固定 `1024` 线程策略被改为 shape-aware 策略：

| `dim_size` | 每个 block 的线程数 | `block_x` | 每个 block 的逻辑 row 数 |
| --- | ---: | ---: | ---: |
| `512` | `512` | `32` | `16` |
| `1024` | `256` | `32` | `8` |
| 其他 persistent shape | `1024` | `min(next_pow2(dim_size), 32)` | 随 shape 变化 |

`512/1024` 也被纳入物理 `grid.x <= 64` 的范围。其思路是：row 越长，单个
warp 持有的元素越多，继续堆叠过多 warp 会放大寄存器和片上存储压力；通过
减少每 block 的 row 数，在吞吐和单 block 资源之间重新平衡。

### 4.5 优化 fast row 的访问与寄存器复用（`03695a9`）

**已实现。** fast row 路径增加了三层优化：

1. 标量路径从 `ILP=1` 提升到 `ILP=4`，归约和写回循环按每线程四个元素
   展开。
2. fp16 对齐 row 继续使用 x4 粒度；对于不能按四元素对齐的 row，增加
   带首元素偏移处理的 `half2` x2 路径，减少落回纯标量路径的 shape。
3. 当 `ceil(dim_size / 1024)` 位于 `2..8` 时，增加寄存器缓存变体。输入
   只从 GM 读取一次，max、sum 和输出阶段复用寄存器中的原始元素。

这一步对应 PyTorch CUDA 中的 ILP/vectorized load 与 register softmax
思路。它优化的是大于 `2048`、但每线程仍能以较小固定数组保存元素的 row。

### 4.6 隔离 `1024` 专用 persistent 编译单元（`5a565c7`）

**已实现。** 当时包含通用 `1024-thread` persistent 模板实例的代码被移入
独立的 `persistent_1024.asc`，构建流程也从“一次编译所有源文件”改为
“逐源文件生成 object，再统一链接”。这里的 `1024-thread` 指该提交时的
线程模板参数，不是 `dim_size=1024`；后续 `bb1c443` 才把该文件收敛为
`dim_size=1024`、VF 使用 `256` 线程的专用流水。

这不是算法优化，而是编译稳定性修复。仓库中的 compiler repro 表明：
`256` 与未执行的 `1024` 模板实例处于同一 ASC 翻译单元时，运行期可能报告
`The configured UB size exceeds 224 KB`；拆分翻译单元后相同逻辑可以运行。

已观察到的事实是：在当前编译器上，模板实例是否位于同一翻译单元可能影响
资源估算。维护上应继续把翻译单元布局视为真实设备可运行性的验证项，不能
只检查编译是否成功。

### 4.7 隔离 `512` 专用 persistent 编译单元（`ea0eeab`）

**已实现。** 在 `1024` 之后，`512` 实例也被移入独立的
`persistent_512.asc`，并为 fp16/fp32 提供显式 dispatch 入口。

`5a565c7` 与 `ea0eeab` 一起建立了后续专用流水的代码边界：高频 shape
可以单独改变线程数、UB 布局和流水，而不扩大通用 persistent 模板的资源
组合。仓库的最小复现直接证明的是 1024-thread 模板同翻译单元问题；目前
没有同等证据表明 512 单独触发过相同错误，因此这里应把 512 拆分理解为
预防性隔离和后续专用流水的工程边界，而不是已有 512 故障的直接修复。

### 4.8 为 `512/1024` 引入 GM/UB 双缓冲流水（`bb1c443`）

**已实现。** 两个专用 kernel 从“SIMT 线程直接反复访问 GM”改为混合流水：

```text
GM input
  -> MTE2 搬入双缓冲 input UB
  -> SIMT VF 在 UB 上完成 max / exp-sum / normalize
  -> MTE3 将 output UB 写回 GM
```

实现使用两个 UB slot 和两组 event id，在连续 tile 之间协调 MTE2、V、MTE3
流水。当前 bucket padded-UB 版本中 `512` 路径每 tile 处理 `32` 行，
`1024` 路径每 tile 处理 `8` 行。

这一优化把 SIMT 线程的离散 GM 访问改成整 tile DMA，并为搬运与计算重叠
创造条件；它不意味着输入输出的总 GM 字节数一定减少。相比简单扩大线程数，
这种实现更贴近 Ascend 的存储层次和流水执行模型。

### 4.9 修复非 `512/1024` persistent fallback（`15efd0e`）

**已实现。** 此前 default 分支会错误调用只接受精确 `dim_size=1024` 的
专用函数，导致其他 persistent shape 进入不匹配的实现。修复后的分发为：

```text
dim_size < 32              -> 通用 1024-thread persistent 模板
32 <= dim_size <= 128      -> persistent_128 bucket padded-UB
129 <= dim_size <= 256     -> persistent_256 bucket padded-UB
257 <= dim_size <= 512     -> persistent_512 bucket padded-UB
dim_size == 1024           -> persistent_1024 专用流水
其他                       -> 通用 1024-thread persistent 模板
```

该提交还把原来的单个大文件拆分为：

- `row_fast.asc`
- `row_persistent_fallback.asc`
- `spatial_kernel.asc`
- `persistent_512.asc`
- `persistent_1024.asc`

拆分后的文件边界与分发路径一致，降低了模板相互影响和后续局部优化的风险。

### 4.10 为可放入 UB 的大 logits row 增加 whole-row 路径（`af01f6f`）

**已实现。** 对常见于 logits 的大 row，新增 whole-row UB recompute 路径：

- fp16：`8192 <= dim_size <= 32768`；
- fp32：`8192 <= dim_size <= 16384`。

一整行先同步搬入 UB，SIMT VF 在 UB 上计算 max、sum 和输出，再整体写回 GM。
为了不在 UB 中额外保存 exp，该路径在 sum 和输出阶段各计算一次 exp，因此
称为 recompute path。

该方案用额外计算换取更少、更规则的 GM 访问，适合 row 能完整放入 UB 的
区间。它也说明阈值不能只看 `dim_size`，还必须考虑输入和输出 dtype 对 UB
容量的共同占用。

### 4.11 为超大 logits row 增加 GM workspace 路径（`18059f8`，已被后续实现替代）

**历史实现。** 当一整行超过当时的 UB 上限时，V3 不再尝试 whole-row staging，而是
拆成三个有全局顺序保证的 kernel：

```text
kernel 1: 计算每行 max     -> row_max[outer_size]
kernel 2: 计算每行 inv_sum -> row_inv_sum[outer_size]
kernel 3: 读取输入、max、inv_sum，写出结果
```

fp16 的触发条件是 `dim_size > 32768`，fp32 是 `dim_size > 16384`。workspace
只按 row 保存两个 float 标量，而不是保存整行中间结果，因此空间复杂度为
`O(outer_size)`。

这个设计牺牲两次额外 kernel launch 和 workspace 分配，换取无需跨 core
同步的全局阶段边界，并继续复用 ILP=4、fp16 x4/x2 和标量访问助手。

### 4.12 释放 224 KiB UB 并统一大 row MTE 分块路径

当前构建为每个 ASC 翻译单元增加：

```text
--cce-disable-vf-stack-reserved-ubuf
--cce-disable-asc-reserved-ubuf
```

在已验证的 Ascend 950PR、CANN 9.2 环境中，可用 UB 从 216 KiB 提高到
224 KiB。实现只把 220 KiB 用于双缓冲数据区，剩余约 4 KiB 留给 reduction
scratch、在线统计、对齐和编译器开销。对应上限为：

- fp16：`56320` 元素；
- fp32：`28160` 元素。

超过整行上限后，FP32 使用两 kernel 流水：

```text
kernel 1: MTE2 分块 GM -> UB，SIMT 在线合并 tile max/sum -> 行级 workspace
kernel 2: MTE2 分块 GM -> UB，SIMT normalize，MTE3 分块 UB -> GM
```

bulk 输入和输出不再由 SIMT 线程直接访问 GM。GM workspace 仍保存每行的
`row_max` 和 `row_inv_sum`，共 8 字节；它用于两个 kernel 之间的全局阶段边界，
不是大 tensor 搬运路径。

FP16 原先也使用同样的两 kernel 拆分，后续已由 4.14 的单 kernel 融合路径
替代。FP32 仍保留这一实现作为 fallback。

在 Ascend 950PR、CANN 9.2、1650 MHz 上用仓库标准 `msopprof BasicInfo`
验证 `(1024, 49152)` fp32：历史 direct-GM 三 kernel 合计 `0.853744 ms`，
当前 MTE/UB 两 kernel 合计 `0.436659 ms`，约为 `1.95x`。FP16 已发布大 shape
在融合前两-kernel 阶段的历史代表性对比基本持平：`longformer_logits` 为
`0.896138 -> 0.898242 ms`，`m2m100_logits` 为 `1.464791 -> 1.464915 ms`。

### 4.13 为 `32 <= dim_size <= 512` 增加 bucket padded-UB 流水

**已实现。** `persistent_128.asc`、`persistent_256.asc` 和
`persistent_512.asc` 现在复用 `persistent_bucket_pad_common.h` 中的 bucket
实现。1024-thread VF 包含 32 个 warp，每个 warp 处理一行；`dim_size` 在
`32..512` 范围内按 `64/128/256/512` bucket 选择固定 UB row stride，GM 到
UB 与 UB 到 GM 都使用高维 stride copy，只在 UB 内保留 padding，不再需要
额外 compact 输出。max、exp-sum 和 normalize 均使用 FP32 累加。小于 32 的
row 保留通用 persistent register/shuffle 路径，避免短 row 的 DMA、event 和
padding 固定成本。

该实现来自 `softmax_bucket_vf_pad` 样例：把非整 bucket 的 row padding
放在 UB stride 中处理，SIMT VF 只根据 `col < dim_size` 屏蔽无效列。没有
被这些 bucket 覆盖的 persistent shape 继续保留原通用 fallback。

### 4.14 融合 FP16 超大 row 的 stats/write 并复用完整 tile

**已实现。** 当 FP16 row 超过 `56320` 元素时，原实现先启动 stats kernel，
再启动 normalize/write kernel。两个阶段 tile 结构相同，但 kernel 边界带来
额外 launch，第二阶段还要重新读取整行输入。

当前实现把每一行的在线 max/sum 统计与 normalize/write 合并到一个 physical
kernel 中，继续使用 1024-thread VF、64 个物理 block、两槽 MTE2/V/MTE3
流水。每行的在线 `(max, sum)` 保存在 `running_stats_ub`，不再分配或访问
`row_max`/`row_inv_sum` GM workspace。统计阶段先读取 tail tile，随后按正序
处理完整 tile，使最后一个 stats slot 保留一块 `56320` 元素的完整 FP16
tile。得到最终统计量后，这一 tile 直接在 UB 中 normalize 并写回，避免每行
从 GM 重读一块完整 tile；其他 tile 再通过原双槽流水完成写回。

FP32 没有复用这一路径：其 tile 宽度、UB 占用和已有收益证据不同，仍走 4.12
的两 kernel fallback。这个 dtype 分流是受控 A/B 的保守边界，不代表 FP32
融合在其他 shape 或工具链上一定无收益。

## 5. 已实现优化方法归纳

| 优化维度 | V3 做法 | 对应 commit |
| --- | --- | --- |
| 版本隔离 | 从 V2 建立独立 V3 实验分支 | `0c24a09` |
| block 并行度 | 先扩大到 1024，再按 512/1024 shape 收缩 | `6c7d930`, `7125e85` |
| grid 并行度 | grid-stride 覆盖逻辑 row，物理 grid 对部分 shape 限制为 64 | `8bffa96`, `7125e85` |
| 混合执行模型 | `__vector__` 外层 kernel + `asc_vf_call` SIMT VF | `8bffa96` |
| ILP | fast row 标量循环按 4 元素展开 | `03695a9` |
| 向量访问 | fp16 x4 之外补充带偏移的 `half2` x2 | `03695a9` |
| 寄存器驻留 | 每线程 2..8 个元素时缓存输入 | `03695a9` |
| 编译器隔离 | 512/1024 模板实例拆分翻译单元 | `5a565c7`, `ea0eeab` |
| UB staging | 512/1024 使用双缓冲 GM/UB 流水 | `bb1c443` |
| exact-128 UB staging | 每 warp 一行、每 tile 32 行的双缓冲 GM/UB 流水 | `17b38fb` |
| 分发正确性 | 专用 512/1024 与通用 fallback 明确分离 | `15efd0e` |
| 大 row 片上计算 | 能放入 UB 时 whole-row recompute | `af01f6f` |
| FP16 超大 row 融合 | stats/write 单 kernel + 完整 UB tile 复用 + UB 在线统计 | 当前变更 |
| FP32 超大 row 分阶段 | MTE/UB tiled stats/write 两 kernel + 行级 workspace | `18059f8` 后续优化 |

## 6. 发布数据中的代表性变化

仓库保存了多次 V3 realistic fp16 发布快照。下面只选择与提交路径直接相关的
shape，并比较相邻阶段。数字是端到端 `latency_ms`。

需要注意：这些快照不是严格控制变量的单 commit A/B，期间可能包含构建、
框架和测量变化。因此它们适合验证趋势，不适合单独证明某行代码的收益。

### 6.1 `512/1024` persistent 专用流水阶段

比较 2026-07-12 快照 `34b4b3e` 与 2026-07-15 快照 `4934a05`。期间主要
实现变化包括 `bb1c443` 和 `15efd0e`。

| case | `dim_size` | 07-12 | 07-15 | 约合加速 |
| --- | ---: | ---: | ---: | ---: |
| `t5_attention` | `1024` | 0.476147 | 0.200236 | 2.38x |
| `convbert_attention` | `512` | 0.195139 | 0.093173 | 2.09x |
| `plbart_attention` | `1024` | 1.465387 | 0.610610 | 2.40x |

数据与专用 GM/UB 流水的优化目标一致：`512/1024` attention shape 获得了
约 2 倍以上的阶段性提升。

### 6.2 历史超大 logits GM workspace 阶段

比较 2026-07-15 快照 `4934a05` 与 2026-07-25 快照 `9f5b1bd`。期间主要
实现变化包括 `af01f6f` 和 `18059f8`。

| case | `dim_size` | 07-15 | 07-25 | 约合加速 |
| --- | ---: | ---: | ---: | ---: |
| `longformer_logits` | `50265` | 1.771074 | 0.967272 | 1.83x |
| `plbart_logits` | `50005` | 3.537761 | 1.915516 | 1.85x |
| `m2m100_logits` | `128112` | 1.694529 | 0.826150 | 2.05x |
| `mt5_logits` | `250112` | 3.151859 | 1.548414 | 2.04x |
| `xglm_logits` | `256008` | 1.706247 | 0.804254 | 2.12x |

这些 case 当时都超过 fp16 whole-row UB 上限，因此进入三 kernel GM
workspace 路径。阶段性收益集中在约 1.8x 到 2.1x；当前实现已改为两 kernel
MTE/UB tiled 路径，不能用这组历史数据代表当前 kernel 拆分。

whole-row 区间的快照结果不完全一致：`convbert_logits` 从 2.277790 ms 降至
2.132864 ms，而 `t5_logits` 从 0.952573 ms 上升至 1.113437 ms。这说明
当前数据提示 `8192` 和 UB 容量上限需要受控 A/B 重新标定；仅凭这些混合
快照，还不能证明当前阈值或 whole-row 路径本身导致了变化。

### 6.3 `dim_size=128` 受控 A/B

在 Ascend 950PR、CANN 9.2.0、Bisheng 15.0.5、1650 MHz 上，使用 seed-0
prepared input 和 CannBench 默认 `BasicInfo` 参数做了两组成对采集。表中为
第 2 组 baseline/candidate；第 1 组同样全部改善。

| case | outer rows | baseline (us) | exact-128 (us) | 改善 |
| --- | ---: | ---: | ---: | ---: |
| `bert_pytorch_attention` | 24,576 | 19.987 | 11.187 | 44.0% |
| `gptj_attention` | 2,048 | 5.396 | 4.132 | 23.4% |
| `gptneo_attention` | 65,536 | 49.090 | 24.407 | 50.3% |
| `mobilebert_attention` | 65,536 | 48.632 | 24.383 | 49.9% |
| `pegasus_attention` | 65,536 | 48.412 | 24.486 | 49.4% |

五个 case 每次都只选择一个 Softmax kernel，current/rated frequency 均为
1650/1650 MHz。最终候选通过 40/40 个 FP16 accuracy case、FP32 `(63,128)`
和 FP16 `(65,128)` row-tail 检查。

### 6.4 FP16 超大 logits 融合与 tile 复用的受控 A/B

在同一 Ascend 950PR、CANN 9.2.0、Bisheng 15.0.5、`dav-3510`、1650 MHz
环境中，使用相同 seed-0 prepared input 和 CannBench 默认 `BasicInfo` 参数
完成两组成对采集。没有传入 warmup、iterations 或 AIC metrics 覆盖。表中
同时列出两组结果，改善按各组自己的 baseline 计算。

| case | shape | baseline R1 (us) | candidate R1 (us) | 改善 R1 | baseline R2 (us) | candidate R2 (us) | 改善 R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m2m100_logits` | `(2048, 128112)` | 1256.064 | 943.492 | 24.89% | 1256.522 | 940.441 | 25.16% |
| `mt5_logits` | `(2048, 250112)` | 2444.042 | 1734.864 | 29.02% | 2445.533 | 1732.471 | 29.16% |
| `xglm_logits` | `(1024, 256008)` | 1232.719 | 913.192 | 25.92% | 1228.231 | 911.581 | 25.78% |

候选对 FP16 `(2,56321)`、`(65,128112)`、`(64,250112)`、`(63,256008)`
以及 FP32 fallback `(4,28161)`、`(3,49152)` 均通过 `torch.softmax` 对比，
最大绝对误差不超过 `1.19e-7`，且没有 NaN/Inf。发布数据采用第 2 组 candidate。

## 7. 当前局限与风险

### 7.1 分发阈值仍以经验常量为主

`64`、`1024`、`8192`、`28160`、`56320` 等阈值直接写在源码中。它们已经
形成可工作的策略，但还缺少由 device 属性、UB 可用容量和 occupancy
统一推导的 cost model。

### 7.2 专用 UB 流水只覆盖精确 `128/512/1024`

其他 persistent shape 仍走通用 GM register/shuffle 路径。例如 196、256
等 realistic shape 的 row 数很多，也可能从小 row tile + 搬运流水中
受益。

### 7.3 whole-row recompute 的收益依赖 shape

把整行搬入 UB 会减少 GM 访问，但同步 copy、较大的静态 UB 数组和重复 exp
也有成本。混合快照提示该区间值得进一步分析，但是否由该路径导致、在哪些
shape 上稳定获益，仍需同环境的受控 A/B 才能确认。

### 7.4 FP32 超大 row 路径仍有 workspace 分配成本

FP16 已融合为一个 kernel，行级统计量驻留在 UB，不再创建 metadata tensor。
FP32 仍会创建 `row_max` 与 `row_inv_sum` 两个 tensor，并保留两个 kernel。
当 `outer_size` 较小或 row 仅略超 UB 上限时，FP32 的分配和 launch 固定开销
仍可能抵消访存收益。

### 7.5 编译器行为是实现约束的一部分

拆分翻译单元解决了已知 UB 资源估算问题，但新增模板变体或重新合并源文件
仍可能使问题复现。编译成功不足以作为验证，必须包含真实设备运行。

### 7.6 当前文档没有严格的逐提交性能实验

发布快照能说明阶段性趋势，但无法隔离单个 commit。后续优化应保存基线、
候选和回退版本在同一环境、同一 warmup/iters 配置下的结果。

## 8. 后续优化方向

以下均为**后续建议**，不表示当前 V3 已经实现。

### P0-1：先建立按路径分组的稳定测量闭环

为 persistent 512、persistent 1024、其他 persistent、register fast、
whole-row UB、large-row tiled 和 spatial 各选代表 case。每次修改至少记录：

- kernel 名和实际命中的分发路径；
- device kernel latency 与端到端 latency；
- GM bandwidth、vector core 利用率、UB 使用量和 occupancy；
- fp16/fp32 精度、row sum、NaN/Inf；
- 全部 40 个 smoke/realistic/stress case 的回归结果。

没有这一步，继续移动阈值很容易把收益从一个 shape 转移成另一个 shape 的
回退。

该项是所有阈值调整的前置条件。完成标准至少包括：固定硬件、构建产物、
warmup/iters 和输入种子；每个候选保留不少于 20 个有效性能样本并报告
median、p95 和离散程度；上述路径代表 case 可确认实际 kernel；40 个
canonical case 全部通过准确率回归。

### P0-2：重新标定 whole-row UB 与 tiled workspace 的边界

当前边界只由 dtype 和 row length 决定。建议至少把以下变量纳入决策：

```text
estimated_cost =
    kernel_launch_cost
  + bytes_moved / effective_bandwidth
  + exp_count / effective_exp_throughput
  + occupancy_penalty(ub_bytes, registers, block_threads)
```

先离线扫描候选阈值，再固化少量分段规则。重点检查 `t5_logits` 这类 whole-row
路径回退的 case，确认问题来自 copy、UB 压力、重复 exp 还是 block/grid
配置。只有 P0-1 的测量闭环就绪后才开始该项；完成时应给出阈值扫描表、
选择依据、已知例外和全部回归结果，而不是只提交新的经验常量。

### P1：扩展 persistent UB tile 的 shape 覆盖

可以把 `512/1024` 的精确模板推广为少量 bucket，例如 128、256、512、1024，
但不要为每个 `dim_size` 生成一个实例。每个 bucket 使用固定 tile 容量，
通过边界判断处理不足部分。

这样既能复用双缓冲流水，也能控制编译时间、二进制体积和翻译单元资源组合。
新增 bucket 仍应保持独立编译或至少按资源规模分组，以防编译器 UB 估算问题
回归。

### P1：减少 FP32 large-row metadata workspace

FP16 已把 stats 和 write 合并为一个 physical kernel，并让最终统计量驻留在
UB。FP32 仍通过 GM `row_max`/`row_inv_sum` 连接两个 kernel。FP32 是否适合
相同融合必须单独 A/B，不能从 FP16 的收益直接推断；候选还必须保持稳定归约
和 MTE2/V/MTE3 的事件顺序。

### P1：减少临时 tensor 分配开销

FP32 的 `row_max` 和 `row_inv_sum` 都只需要 `outer_size` 个 float。可以评估：

- 合并为一个 `[outer_size, 2]` workspace；
- 通过 operator-local workspace 管理复用分配；
- 在 `outer_size` 很小时直接回退单 kernel，避免 workspace 路径。

任何复用方案都必须保持 stream 安全，不能在并发调用之间共享未隔离的裸
buffer。

### P1：继续优化 fast path 的对齐和寄存器压力

现有 register-cache 只按每线程元素数 `2..8` 判断。后续可结合 dtype、
实际寄存器数和 occupancy，把它改成更精细的选择，并分别测量：

- x4、shifted x2、scalar 三种访问的有效带宽；
- 缓存原始输入与缓存 exp 的取舍；
- `1024` 线程 block 是否始终优于 `512` 线程 + 更高 occupancy。

### P2：补充 spatial path 的针对性优化

V3 的主要优化集中在 `inner_size == 1`，spatial path 基本沿用 CUDA-like
形态。后续可以针对 `tiny_channel_softmax` 和
`channelwise_activation_map` 等覆盖 case，评估：

- `block_x/block_y` 与物理一维 grid 展平策略；
- `block_x == 1` 的串行归约是否值得专用向量实现；
- 多个 inner position 的连续搬运和 UB tile 化。

spatial case 数量较少，优先级应低于 realistic 数据中占主导的 row-wise
路径。

### P2：把 launch policy 收敛为 operator-local cost model

长期可以把下列输入统一交给 V3 operator-local policy：

```text
(dtype, outer_size, dim_size, inner_size, alignment, device_properties)
    -> path, block shape, grid cap, UB tile, workspace size
```

这个 cost model 应继续留在 Softmax 插件目录内，不应把具体 shape 或
Softmax 名称硬编码到公共 backend。

## 9. 验证建议

### 9.1 正确性

从仓库根目录执行已有脚本，覆盖 40 个 canonical case：

```bash
python src/cannbench/operators/builtin/softmax/simt/test/ascend_softmax_accuracy.py \
  --mode both \
  --dataset ALL \
  --case ALL \
  --dtype float16 \
  --warmup 0 \
  --iters 1
```

对改变累加或归约顺序的优化，还应增加 fp32、极端 logits、非四元素对齐和
row sum 检查。

### 9.2 路径覆盖

至少确认以下路径各有一个 case：

- 通用 persistent；
- `512` 专用 persistent；
- `1024` 专用 persistent；
- fast register-cache；
- FP16 large-row 单 kernel MTE/UB tiled stats/write；
- FP32 large-row 两 kernel MTE/UB tiled stats/write；
- whole-row UB recompute；
- spatial。

### 9.3 编译器回归

涉及模板实例、launch bounds 或翻译单元调整时，运行
`../test/compiler_repro/` 中的 single-TU 与 split-TU 对照，并在真实设备上
执行目标 shape。尤其不能因为未执行的模板实例“看起来无关”就忽略它对资源
估算的影响。

## 10. Commit 索引

| 日期 | Commit | 作用 |
| --- | --- | --- |
| 2026-07-03 | `0c24a09` | 建立 V3 独立基线 |
| 2026-07-03 | `6c7d930` | persistent/fast 扩大到 1024 线程 |
| 2026-07-04 | `8bffa96` | 混合 SIMD/SIMT launch 与 shape-aware grid |
| 2026-07-04 | `7125e85` | 512/1024 的 shape-aware block 策略 |
| 2026-07-07 | `9258b14` | 目录迁移到 builtin operator package，无算法变化 |
| 2026-07-09 | `03695a9` | ILP、shifted half2、register-cache fast path |
| 2026-07-11 | `5a565c7` | 隔离通用 1024-thread persistent 模板 |
| 2026-07-11 | `ea0eeab` | 隔离 512-thread persistent 模板 |
| 2026-07-14 | `bb1c443` | 512/1024 GM/UB 双缓冲流水 |
| 2026-07-14 | `15efd0e` | 修复 persistent fallback 并拆分源文件 |
| 2026-07-25 | `af01f6f` | whole-row UB logits 路径 |
| 2026-07-25 | `18059f8` | 超大 logits 三 kernel GM workspace 路径 |
| 2026-08-11 | `17b38fb` | exact-128 单行 GM/UB 流水 |
| 2026-08-11 | 当前变更 | FP16 超大 logits 单 kernel 融合与完整 UB tile 复用 |

补充证据：`cebbb3d` 添加了 persistent 单翻译单元与拆分翻译单元的编译器
最小复现。它不属于 V3 目录本身的提交历史，直接支持 `5a565c7` 的问题
分析，也为 `ea0eeab` 的预防性隔离提供了工程背景。

## 11. 参考资料

- [Softmax Dispatch Shape Notes](../README.zh-CN.md)：PyTorch CUDA 与
  Ascend SIMT V1/V2 的 shape 分发对照。
- [Issue 002](../../../../../../../docs/issues/002-ascend-simt-softmax-row-wise-v2.md)：
  V1 correctness 问题、V2 分发演进和准确率验证记录。
- [Softmax V3 Compiler Repro](../test/compiler_repro/README.md)：persistent
  多模板实例的编译器最小复现。
- [V3 realistic fp16 发布数据](../../../../../../../published/opbench-ascend-950pr-simt-v3-softmax-realistic-float16/meta/benchmark-records.json)。
- [Softmax Huge Logits Fused Pipeline Design](../../../../../../../docs/designs/softmax-huge-logits-fused-pipeline-design.md)。
- PyTorch CUDA `aten/src/ATen/native/cuda/SoftMax.cu`。
- PyTorch CUDA `aten/src/ATen/native/cuda/PersistentSoftmax.cuh`。

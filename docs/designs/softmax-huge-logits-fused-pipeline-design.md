# Softmax Huge Logits Fused Pipeline Design

## Goal

Reduce Ascend 950PR Softmax V3 latency for huge FP16 rows that exceed the
whole-row UB limit. The target realistic family is:

| Case | Shape | Baseline R2 |
| --- | --- | ---: |
| `m2m100_logits` | `(2048, 128112)` | 1256.522 us |
| `mt5_logits` | `(2048, 250112)` | 2445.533 us |
| `xglm_logits` | `(1024, 256008)` | 1228.231 us |

The implementation remains operator-local, preserves stable FP32 accumulation,
and uses only the C API, Tensor API, and SIMT API boundary allowed by the
repository.

## Bottleneck

The original huge-row path launched two kernels per operator call:

```text
kernel 1: tiled GM -> UB, online max/sum -> row workspace
kernel 2: tiled GM -> UB, normalize, UB -> GM
```

This structure gave a simple global stage boundary, but it incurred two costs
for every row: an extra physical launch and a complete second read of the input.
The three realistic FP16 shapes spend enough time in the tiled path for those
costs to be material.

## Selected Architecture

Fuse tiled stats and normalize/write into one physical kernel for FP16 huge
rows. Each physical block owns independent rows through a grid-stride loop, so
the fusion requires no inter-core synchronization.

The retained pipeline uses:

- 64 physical blocks and a 1024-thread SIMT VF;
- two `56320`-element FP16 UB slots;
- the existing MTE2/V/MTE3 two-slot event protocol;
- online stable `(max, sum)` combination with FP32 accumulation;
- a two-float `running_stats_ub` state for the current row, without GM metadata
  workspace or temporary metadata tensors;
- the existing two-kernel implementation as the FP32 fallback.

The stats traversal handles the tail tile first, then visits complete tiles in
ascending order. The final stats slot therefore retains one complete tile in
UB. After the final row statistics are available, that retained tile is
normalized and written directly, avoiding one `56320 * sizeof(fp16)` GM read
per row. Remaining tiles use the same two-slot load, VF, and write pipeline.

```text
tail tile -> stats
full tiles -> stats, retain final full tile in UB
retained tile -> normalize/write without GM reload
remaining tiles -> pipelined GM reload, normalize/write
```

## Dispatch Boundary

`dispatch_row_fast_large_tiled_fused_forward_kernel` handles the fused FP16
implementation. The FP32 path retains the established two-kernel stats/write
implementation and its per-row GM workspace. Shapes at or below the whole-row
UB limit continue to use the existing whole-row path, and persistent or spatial
dispatch is unchanged.

This dtype boundary is deliberate. The measured gain applies to the tested
FP16 shapes and tile geometry; it is not evidence that FP32 has the same UB,
traffic, or compiler tradeoff.

## Alternatives

The first fused candidate removed the launch boundary but reloaded every tile
during write. On `m2m100_logits`, it reduced latency from 1256.064 us to
967.454 us, a 23.0% improvement.

The retained candidate additionally reused one complete UB tile. Its first
discrimination run measured 940.071 us for `m2m100_logits` and 911.800 us for
`xglm_logits`, so it was selected for full paired validation.

Keeping both stages as separate kernels was retained only for FP32. It remains
the conservative fallback where fused-path correctness and performance have
not been established.

## Correctness

The candidate was compared with `torch.softmax` on the target device. It
passed representative limit, tail, multicore, and FP32 fallback shapes:

| Dtype | Shape | Max absolute error |
| --- | --- | ---: |
| FP16 | `(2, 56321)` | `0` |
| FP16 | `(65, 128112)` | `5.96e-8` |
| FP16 | `(64, 250112)` | `1.19e-7` |
| FP16 | `(63, 256008)` | `5.96e-8` |
| FP32 | `(4, 28161)` | `1.16e-10` |
| FP32 | `(3, 49152)` | `5.82e-11` |

No comparison produced NaN or Inf.

## Paired Performance

Both pairs used `Ascend950PR_9589`, CANN 9.2.0, Bisheng 15.0.5, `dav-3510`,
64 blocks, 1650/1650 MHz, the same seed-0 prepared inputs, and CannBench default
`BasicInfo` parameters. No warmup, iteration, or AIC metric override was used.

| Case | Baseline R1 | Candidate R1 | Gain R1 | Baseline R2 | Candidate R2 | Gain R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m2m100_logits` | 1256.064 us | 943.492 us | 24.89% | 1256.522 us | 940.441 us | 25.16% |
| `mt5_logits` | 2444.042 us | 1734.864 us | 29.02% | 2445.533 us | 1732.471 us | 29.16% |
| `xglm_logits` | 1232.719 us | 913.192 us | 25.92% | 1228.231 us | 911.581 us | 25.78% |

Published records use the candidate R2 values.

## Provenance

- baseline remote root:
  `/root/cannbench-softmax-dim128-single-row-jFxg8x`;
- candidate remote root:
  `/root/cannbench-softmax-huge-reuse-a1-UuQ14J`;
- baseline extension SHA-256:
  `4d3c14450189a3c0e36aecf02eac9a96bea67ea7ce81f4cf035500c6cd14bf49`;
- candidate extension SHA-256:
  `44abf45762955a975b480ad60fc113fd2c3258af1f2011b70c23d9f0779bbba2`;
- candidate source SHA-256:
  `d3b30e576fed1782dc95589f4a3940fe60e0deebd54a0cfcf13ec2f786524857`;
- baseline R1 results:
  `/tmp/cannbench-softmax-huge-baseline-r0-DtWvyP/softmax-huge-baseline-r0`;
- candidate R1 results:
  `/tmp/cannbench-softmax-huge-reuse-a1-r1-PdDyNL/softmax-huge-reuse-a1-r1`;
- baseline R2 results:
  `/tmp/cannbench-softmax-huge-baseline-r2-0vosTw/softmax-huge-baseline-r2`;
- candidate R2 results:
  `/tmp/cannbench-softmax-huge-reuse-a1-r2-sEFmxC/softmax-huge-reuse-a1-r2`.

The result is specific to the recorded 950PR variant, software stack, FP16
shapes, and BasicInfo device timing boundary. FP32 fusion, other 950 variants,
and rows near the whole-row dispatch threshold remain untested.

## Publication

Only the latency fields for `m2m100_logits`, `mt5_logits`, and `xglm_logits`
are updated. Record ordering, run IDs, accuracy, schemas, and all unrelated
published data remain unchanged.

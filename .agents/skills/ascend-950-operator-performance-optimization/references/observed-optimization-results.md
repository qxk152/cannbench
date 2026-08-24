# Observed Performance Patterns

## Contents

- [Purpose And Evidence Boundary](#purpose-and-evidence-boundary)
- [Algorithm And Work Mapping](#algorithm-and-work-mapping)
- [Tiling, Lifetime, And Fusion](#tiling-lifetime-and-fusion)
- [Producer Consumer Pipelines](#producer-consumer-pipelines)
- [Layout And UB Conflict Handling](#layout-and-ub-conflict-handling)
- [Packed Types And Vector Width](#packed-types-and-vector-width)
- [Row Reduction Shape Specialization](#row-reduction-shape-specialization)
- [Resource And Launch Geometry](#resource-and-launch-geometry)
- [Measurement And Attribution Failures](#measurement-and-attribution-failures)
- [How To Apply These Results](#how-to-apply-these-results)

## Purpose And Evidence Boundary

Use this reference to form priors, avoid repeating disproven experiments, and
set realistic retention gates. It is not a performance guarantee.

Treat these results as Ascend 950-family evidence, primarily Ascend 950PR. Do
not transfer an exact gain, hardware limit, synchronization property, or
compiler behavior to another Ascend generation; remeasure even when the source
transformation looks identical.

The initial synthesis covered the complete workspace Codex session corpus
through 2026-08-05: 43 logical sessions reconstructed from 154 relevant rollout
files. An incremental exact-working-directory scan through 2026-08-10 raised
the corpus to 163 rollout files across `cannbench` and `cannbench-2`. Repository
notes, source, commits, raw benchmark records, and profiler artifacts were used
only to clarify or cross-check session claims.

Most multi-stage, latency-oriented measurements used Ascend 950PR
(`dav-3510`), CANN 9.2.0, production `-O3`, 1,650 MHz, BF16, and this
shape family:

```text
batch=2, query rows=2, context rows=32768, groups=64
feature width=128, selected rows=2048
consumer groups=128, secondary feature widths=576 and 512
```

Row-reduction measurements used the same device family and toolchain, but dtype,
row width, outer rows, and timing method vary by row below.

Positive percentages mean lower latency. Kernel, component, and workflow are
different boundaries. Do not add isolated percentages or compare rows with
different baselines as if they were one cumulative experiment.

## Algorithm And Work Mapping

| General technique and scenario | Observed result | Reusable conclusion |
| --- | --- | --- |
| Replace repeated `Top-K + 128` padded 4096-element bitonic sorts in a many-query path | Selection stage `3134.926 ms` versus an optimized-library `5.929 ms`, a `528.7x` gap; about 985,088 merges and 76.84 million synchronization stages | Fix repeated asymptotic work before tuning instructions, tiles, or launch geometry. |
| Map one 32-lane warp to each 32-position tile entry and reduce 64 channels across lanes | Producer kernel `292.150 -> 147.763 us` (`49.4%`); component `34.1%`; workflow `17.2%` | Fill already-launched threads with independent reduction work before changing thread count. |
| Apply the same group-parallel reduction to a many-query full-score path | Producer `276.019 -> 136.839/136.848 ms` (`50.4%`); component `46.4%`; workflow `21.6%` | A work mapping can transfer across workload classes when dtype, tile shape, and ownership match. |
| Change only VF threads from 1024 to 32/64/128/256 while retaining a serial 64-channel loop | Best producer gain `1.8%`; best workflow gain `0.8%` | Removing inactive threads is not equivalent to parallelizing the dominant loop. |
| Replace atomic and repeated threshold-compaction scans with one packed-count scan | Selection chain `37.8%-37.9%`; component `22.9%-23.1%`; workflow `10.0%-10.1%` | Once a threshold is known, scan each candidate once and derive deterministic offsets. |
| Partition a length-32768 context into 16 shards and reduce shard-local histograms | Complete selection chain about `82.031 -> 56.759-57.279 us` (`30.2%-30.8%`); workflow `7.85%-7.98%` | Distributed histograms help when context ownership already exists and one-block selection is serial. |
| Fuse five distributed Top-K stage launches into one 64-task kernel with four producer-visibility plus device-wide barrier boundaries | Top-K `35.704-36.238 -> 28.421-28.640 us` (`19.9%-21.0%`); complete Indexer `7.6%-8.5%`; selected workflow `3.6%-4.3%` | Launch-bound stage fusion paid after preserving GM visibility and making all 64 tasks reach every barrier, including the 60 masked reducer tasks. This used a task-local API exception and is not blanket permission to cross the repository API boundary. |
| Parallelize a low-byte reducer's serial shard tails and offset construction | Reducer `28.904 -> 14.423-14.444 us`; workflow `5.56%-5.62%` | Activate idle threads in the measured dominant tail, not merely in an already-small prefix. |
| Group a 256-bin threshold scan | Dominant low scan `14.945 -> 7.494-7.600 us`, workflow `1.95%-2.45%`; smaller high scan improved locally `16.7%-20.2%` but workflow only `0.117%-0.538%` | The same local transformation can be retained or rejected solely because its absolute workflow share differs. |
| Pair two query rows but shrink score tiles, increasing selection merge rounds | `11226.192 -> 12472.903 ms`, `11.11%` slower | Reuse is a loss when it increases an expensive downstream boundary more than it saves upstream. |
| Share one selected-128 KV gather across two Head64 groups using four AIV producers | Published selected workflow `208.847 -> 191.162 us` (`8.47%`) | Reuse irregular source traffic across consumers that share the same KV before tuning the per-lane gather instruction shape. |
| Replace a copied gather with per-row QK and V VF calls plus per-row DCCI for 8,192 selected rows | Sparse main kernel `54.483 -> 882.803-883.764 us` (`16.21x`); selected workflow regressed `40.62%` | A VF is not a free function call. Batch rows inside a VF and publish visibility at the tile or phase boundary instead of paying entry, address-generation, and cache-maintenance cost per row. |

## Tiling, Lifetime, And Fusion

| General technique and scenario | Observed result | Reusable conclusion |
| --- | --- | --- |
| Enlarge a matrix-product tile from 64 to 128 after calculating live L1 state | Fused consumer `11.2%`; consumer component `10.3%`; workflow `6.3%` | Fewer packing and handshake rounds paid while the capacity worksheet retained margin. |
| Restore 32 active warps after a stale four-warp correctness workaround was no longer needed | Fused consumer `13.4%`; workflow `6.8%` | Revalidate old correctness workarounds after synchronization changes. |
| Reuse compact `int32` row offsets across repeated gather phases | Fused consumer `9.0%`; workflow `4.2%` | Persist small metadata when it removes repeated address, mask, and range calculations. |
| Compute four merge weights once per lane and reuse them across 512 output elements | Helper `36.096 -> 12.012 us` and `36.179 -> 12.289 us` (`66.4%`); workflow `3.7%` | A large helper-local gain is capped by that helper's absolute workflow share. |
| Reuse L1 storage across sequential compute phases and enlarge one tile from 128 to 256 | Fused consumer `4.74%-5.94%`; workflow `2.55%-3.10%` | Calculate phase maxima rather than summing buffers that are never live together. |
| Enlarge a second matrix-product tile to 256 while preserving a persistent operand across all selected tiles | Fused consumer `5.61%-6.82%`; workflow `2.74%-3.26%` | A tile change can depend on fixing a lifetime bug rather than on additional memory. |
| Increase selected-token tile from 64 to 128 with phase-local ownership | Fused consumer `18.53%-19.23%`; workflow `9.41%-9.42%` | Removing state-update, copy, and synchronization rounds can dominate larger live state. |
| Use an outer tile of 256 with two streamed 128-row compute subtiles | Workflow `279.771 -> 270.696/271.191 us` (`3.07%-3.24%`) | Streaming subtiles can reduce outer transitions without making the full operand resident. |
| Increase the outer tile to 512 but replay a packed persistent operand three times | Workflow improved only `0.244%-0.674%` | Replayed persistent state can consume most of a larger tile's theoretical gain. |
| Reuse an earlier packed latent operand in a later phase but double outer-loop overhead | One workflow pair improved `0.075%`; another regressed `0.641%` | Removing logical GM reads does not prove less physical traffic; cache and phase overhead can erase the saving. |
| Keep row-major Value data through UB/L1 and transpose during the L1-to-L0B copy | Published selected workflow `219.827 -> 208.847 us` (`5.00%`) | Prefer a supported downstream copy-format conversion when it removes a repeated explicit UB transpose and its intermediate layout. |
| Combine two Value-256 Cube results into one logical PV-512 handoff and one output-update VF | Sparse fused kernel improved about `6.97%`; selected workflow checkpoint improved about `3.61%` | Merge producer handoffs when the two results have one consumer and a common lifetime; the gain came from one fewer ready/free exchange and VF boundary, not less MMAD work. |
| Cache each lane's four scaled softmax values and validity bits across max and exp passes | Sparse fused kernel `10.66%`; selected workflow checkpoint `5.46%` | Lane-local reuse can beat shared UB metadata because it removes repeated loads without adding a block barrier or shared-memory traffic. |
| Move canonical Query packing from an AIV VF to direct AIC GM-to-L1 ND2NZ | Sparse fused kernel `5.37%-5.45%`; selected workflow `2.22%-2.42%` | Deliver data directly in the consumer layout when the source stride is supported and the change removes a VF, UB-to-L1 copy, and CrossCore handoff together. |
| Fuse an FP16 row pipeline for `32768 < width <= 51200` | Scalar fused form regressed about `32%`; after packed normalize/write, three shapes improved `21.21%-25.15%` and aggregate improved `23.12%` | Fusion pays only when the fused inner schedule is also efficient. Always retain an unfused control. |
| Fuse FP16 tiled stats/write for widths `128112-256008`, keep online `(max,sum)` in UB, visit the tail first, and retain one complete `56320`-element tile for normalize/write | Fusion without retention improved one case `23.0%`; the retained candidate improved three cases `24.89%-29.16%` in each of two paired collections and removed the `row_max`/`row_inv_sum` GM workspace | When one block owns a row, phase fusion can remove a launch and GM metadata handoff; tail-first traversal can also preserve a full tile across the phase boundary. Attribute fusion, metadata removal, and reuse separately, and keep untested dtypes on their established fallback. |

## Producer Consumer Pipelines

| General technique and scenario | Observed result | Reusable conclusion |
| --- | --- | --- |
| Two-slot BF16 producer/consumer overlap across 64 repeated tiles | Producer `148.363 -> 82.889 us` (`44.1%`); component `23.7%`; workflow `12.2%` | Double buffering paid because producer and postprocess had exposed, repeated overlap. |
| Initial two-slot implementation reused the same flag IDs for ready and free directions | Device hang; a separate invalid conversion parameter later produced a runtime error, not a timeout | Use distinct directional state and classify synchronization versus instruction-parameter failures separately. |
| Replace an explicit V-to-MTE3 ordering edge with direct PIPE_V publication of PV-free | Kernel completed, but output mismatched `247866/262144` and LSE `512/512` | A cross-core flag is not automatically a substitute for a local pipeline ordering edge. Remove an event only after proving producer completion and visibility to the flag's pipeline. |
| In-place MTE2/compute/MTE3 pipeline for FP16 rowwise `8192 <= width <= 32768` | Three representative shapes improved only `2.59%-3.12%` | A valid pipeline can have small benefit when compute, not transfer, is dominant. |
| Merge max/sum into one UB stats kernel for FP16 `32768 < width <= 51200`, limiting physical grid to 64 | Three shapes improved `25.30%-37.67%` | Reducing scans and launches paid only after each physical block owned enough rows for slot reuse. |
| Tiled online `(max,sum)` for FP16 rows wider than 51200 | Representative and stress shapes improved `17.16%-21.84%` | Online stable statistics work when a full row cannot reside in UB. |
| Tiled MTE2/compute/MTE3 final write for FP16 rows wider than 32768 | Two representative full components improved about `27.3%` and `40.5%`; changed write stage `44%-56%` | Convert direct-GM bulk traffic only after profiling proves it is a dominant stage. |
| UB pipeline for a nine-element row with about six small tiles per core | Median regressed `22.8%` | Event, copy, and VF fixed costs can exceed direct-GM traffic for very short rows. |
| Recreate a three-slot P=1 rolling schedule while retaining duplicate irregular gathers | Fused candidate mean `230.933 us` versus P=4 complete Sparse Attention `105.744 us` (`2.18x` slower), despite removing Combine and partial output | Scheduling resemblance is insufficient when the reference path's decisive data-reuse and copy primitives are missing; fix the traffic pattern before copying its pipeline depth. |

## Layout And UB Conflict Handling

| General technique and scenario | Observed result | Reusable conclusion |
| --- | --- | --- |
| Stage gathered rows contiguously, then transpose 16-by-16 tiles into the required blocked layout | Fused consumer `31.03%-31.27%`; workflow `18.25%-18.54%` | A measured transpose can be much cheaper than repeated scattered stores. |
| Emit a 64-by-32 BF16 producer tile in capacity-neutral column-major form with inline format conversion | Producer `31.89%-32.57%`; workflow `10.60%-11.15%` | Change the producer layout when it removes a repeated eight-way UB conflict without adding a separate pass. |
| Pad adjacent FP32 blocked-layout result planes with 15 gaps of 64 bytes | Changed component `0.74%`; workflow `0.70%` | Small padding can yield a small repeatable gain; use a predeclared gate appropriate to the boundary. |
| Pad a read-side blocked layout by 960 bytes | Changed kernel mean improved only `0.24%`; workflow mean regressed `0.33%` | Reduced conflict can be offset by padded copy and address arithmetic. |
| Remap four producer staging paths into predicted physical order | Workflow deltas `-0.37%`, `-0.20%`, `-0.12%`, and `-3.85%` | Conflict count alone is not an objective; include indexing, store pattern, and transpose cost. |
| Replace a four-way strided reducer read with all-thread loads and four shuffles | Local reducer regressed `1.6%` and `5.6%` | Shuffle count and active-thread cost exceeded the removed conflict. |
| Increase two fixed shared slots from 8 KiB to 9 KiB to add row padding | Runtime UB out-of-bounds before timing | A larger dynamic UB request does not enlarge a separate fixed per-subunit shared region. |
| Reorder FP32 ILP=4 UB access to make each emitted instruction use an 8-byte lane stride | Synthetic whole-row case `19.2%`; tiled total `7.0%` | Analyze one emitted instruction and dtype at a time. The corresponding FP16 path was already conflict-free. |

## Packed Types And Vector Width

| General technique and scenario | Observed result | Reusable conclusion |
| --- | --- | --- |
| Use packed FP16 loads in whole-row and tiled max/sum loops | Controlled shapes improved `13.7%-25.9%`; a fresh 30-case package reduced mean latency `12.07%` | Packed access paid when it reduced useful load and conversion instructions while keeping FP32 accumulation. |
| Use packed BF16 only in one later adjacent-load/store packing boundary | Changed component `2.33%`; fused kernel `2.57%`; workflow `1.32%` | Vectorize concrete boundaries independently; one success does not justify global conversion. |
| Stack the small blocked-layout padding and the successful packed-BF16 boundary | Changed component `3.15%`; fused kernel `3.59%`; workflow `1.39%` | Isolated gains were not additive because an unchanged component moved by `-1.10%`. Measure the combined package. |
| Use packed BF16 in earlier packing, query packing, probability stores, value staging, or post-load arithmetic | Local deltas ranged from a `2.73%` regression to only `0.41%` improvement; workflow was noise or worse | Packed source types do not help if memory transactions are unchanged or values are immediately unpacked for FP32 math. |
| Group adjacent values as 32-bit or 64-bit operations while retaining a scattered destination | Fused kernel regressed `36.6%-37.1%` and about `113%`; workflow regressed `21.7%-21.9%` and about `67%` | Wider operations can make the emitted load/store schedule much worse even with zero Stack and acceptable registers. |

## Row Reduction Shape Specialization

| General technique and scenario | Observed result | Reusable conclusion |
| --- | --- | --- |
| Bucket widths `32-512` into fixed `64/128/256/512` UB row strides, keep padding only in UB, and use strided GM-to-UB and UB-to-GM copies for the logical row bytes | One shared padded-UB pipeline covered non-exact widths without writing padded bytes to GM or launching a separate compact-output pass; its isolated contribution was not measured apart from the combined package | A fixed compile-time UB stride can retain bucketed VF structure without inflating the external layout. Keep the logical GM stride independent from the padded UB stride, mask invalid columns in compute, and retain a generic fallback where DMA, event, or padding cost dominates. |
| Dedicated persistent GM/UB pipelines for exact widths 512 and 1024 | Historical adjacent snapshots showed `2.09x-2.40x` on representative FP16 shapes | Directional evidence only; snapshots were not a controlled single-change A/B. |
| Bucket widths 129-256 into one persistent UB pipeline and pipeline width 1024 separately | Six 129-256 cases aggregate `1950.873 -> 1427.170 us` (`26.84%`); two width-1024 cases `810.992 -> 738.867 us` (`8.89%`) | Bucket by behavior rather than generating one specialization per exact width. |
| Tighten the broad 129-256 persistent bucket to generated-work caps of 160 and 224 | Widths 144-204 across different outer sizes improved repeatably by about `8.5%-31%`; an unchanged neighboring width-256 control was neutral after build-layout isolation | Bucket by generated iterations, active lanes, and resource behavior, then validate several outer-size regimes, both sides of each dispatch boundary, and an unchanged neighboring range. |
| Add a one-row-per-warp persistent GM/UB pipeline for exact width 128 | Five realistic cases improved `23.4%-50.3%`; reusing the width-256 bucket regressed a controlled 65,536-row case from `34.74` to `59.19 us` | A frequent exact width can justify its own specialization when a broader bucket leaves iterations or lanes inactive. Compare exact and bucketed controls across both small and large outer sizes. |
| Add a UB producer/consumer pipeline for widths 33-64 | A small-outer case improved about `9%`, while a large-outer case in the same width bucket regressed `14%-15%`; the candidate was rejected | DMA, event, and VF fixed costs can reverse a bucket's value as outer size changes. One favorable case cannot authorize an entire width bucket; include small- and large-outer controls before retaining it. |
| Replace always-true runtime element guards with compile-time exact-shape bounds and narrow proven tile-local indices to `int32_t` in width-512/1024 VFs | FP16/FP32 registers fell `122 -> 63` / `107 -> 61` for width 1024 and `64 -> 39` / `59 -> 39` for width 512, all zero-spill; seven FP16 cases improved `1.64x-2.76x` | Specialization is incomplete while unrolled loops still depend on opaque runtime bounds. Surface caller invariants to the compiler and right-size local arithmetic, but treat this combined change as one result unless the factors are separately measured. |
| Apply the same compile-time-bound and tile-local-`int32` pattern to an exact width-256 VF while preserving wide GM and outer-row offsets | Controlled FP16 pairs at outer rows 1024, 16384, 65536, and 262144 improved `20.44%`, `37.39%`, `43.80%`, and `33.54%`; width 224/225/255/257 dispatch controls retained their kernel families | A frequent exact width can justify specialization when the invariant removes generated guards across every unrolled iteration. Prove transfer across several outer regimes and both sides of the dispatch boundary; do not dispatch by case or complete shape. |
| Force direct-GM x4, shifted-x2, or scalar paths for FP16 rows around 30K-50K instead of the established whole-row UB path | Direct x4/x2 regressed `88%-116%`; automatic direct regressed `100%-138%`; scalar regressed `268%-292%` at 1650 MHz | Removing DMA/events is not a performance argument by itself. Preserve the UB implementation as a controlled baseline and reject a direct selector before refining thresholds when its first general boundary loses materially. |
| Give one thread all nine row elements | Median regressed `28.7%` | Register lifetime and strided cross-thread access were worse than contiguous subwarp reduction. |
| Increase short-row warp batch from two rows to four | Median regressed `20.6%` | Fewer loop trips did not offset generated register and scheduling cost. |
| Validate the final stacked Softmax package against all 30 published realistic FP16 cases in one standard CannBench round | Versus the previous published package, 29/30 cases improved and geometric-mean speedup was `1.168x`; routing width 9 into padded UB first changed `8.028 -> 14.301 us`, while restoring the generic path produced `8.187 us`. All published accuracy errors were exactly zero and accepted raw rows were `1650/1650 MHz` | Full-workload validation must include every dispatch boundary and preserve a proven generic fallback. A large aggregate gain does not authorize one regressed regime; the observed `<32` cutoff is Softmax-package evidence, not a universal threshold for another operator or stack. |
| Use a fixed-grid specialization for one exact `49152 x 9` shape | Reached about `4.7-5.3 us` but was reverted | A fast exact-shape experiment is not a general dispatch policy. |
| Cache FP16 or FP32 exponentials in UB for 30K-32K rows | FP16 about `28%` slower; FP32 about `22%-23%` slower | Saving exponentiation did not compensate for extra UB traffic and lost overlap. Fit is feasibility, not performance. |
| Release optional UB reservations and generalize a two-kernel tiled path | Synthetic FP32 `(1024,49152)` improved `0.853744 -> 0.436659 ms` (`1.95x`); two measured FP16 cases were effectively neutral | The same memory schedule can be dtype-sensitive. Do not infer FP16 from FP32 or vice versa. |
| Expand gather staging from 16 to 32 rows to remove one UB-to-GM copy and MTE event | Sparse fused mean regressed `1.23%` | Fewer transfers can lose when a larger staging lifetime or copy shape costs more; test transfer granularity rather than maximizing it. |
| Cache tile validity in shared UB with a producer warp and block barrier | One pair improved `0.143 us`, the other regressed `0.547 us`; mean regressed `0.22%` | Shared metadata is a loss when repeated reads are already small and the barrier/shared traffic exceed them. Prefer ownership-local reuse first. |
| Split QK-576 into five QK-128 segments with two L0A/L0B staging slots | Build and source tests passed, but the first device synchronization returned `507015` | More overlap can fail because extra MMAD and event edges enlarge the protocol. Reject a runtime-failing pipeline before layering synchronization fixes on it. |

## Resource And Launch Geometry

| General experiment | Observed result | Reusable conclusion |
| --- | --- | --- |
| `launch_bounds(1024)` with 32 registers and 32 B Stack versus `launch_bounds(512)` with 48 registers and zero Stack | 16-round medians `98.219` and `98.253 us`, effectively equal | Removing a small spill does not guarantee latency improvement. |
| Fixed 16,384-element scan over `4x2048`, `8x2048`, `16x1024`, `32x512`, `64x256` | Medians `11.002`, `6.756`, `4.657`, `3.824`, `4.081 us`; `32x512` beat `64x256` by `6.3%` | Maximum cores or threads was not optimal for this workload. |
| Widen selected packing VFs to 2048 threads | Fused component regressed `1.3%`; workflow `0.6%`, despite 12/13 registers and zero Stack | Resource feasibility is necessary but does not predict useful scaling. |
| Estimate a 2048-thread change from the measured critical-path share before coding | Ideal saving was at most `4.62%` of the fused boundary before overhead | Reject candidates whose physical upper bound cannot clear the retention gate. |
| Move 16 tasks/256 threads to 32 tasks/1024 threads in a many-row path | Selected kernel `11.219 -> 3.125 s` (`3.59x`), but synchronized public boundary `11.226 -> 16.672 s` (`48.5%` slower) | Preserve conflicting measurement boundaries; a kernel win is not an end-to-end win. |
| Compiler loop versus manual expansion | Loop faster about `20.4%` in direct SIMT and `34.2%` in mixed execution; other paired samples showed manual expansion slower `14.2%-25.3%` | Source-level expansion can worsen Stack, registers, or scheduling. Inspect emitted resources and device time. |
| Remove an L1 score handoff and deliver L0C results closer to the consumer | Fused kernel improved only `0.81%-0.87%`; workflow regressed `0.33%` in one pair and improved `0.50%` in the other | Eliminating one logical copy is not enough when the absolute edge is small and neighboring stages move by similar amounts. |
| Combine larger 128-context Indexer packets with two-slot L1 prefetch | Score regressed `3.40%-4.33%`; workflow regressed `0.92%-1.55%` | Larger copies plus a more complex ownership protocol can cost more than the reduced copy count. Do not retry the combined mapping without evidence that one factor changed. |
| Reduce width-197 row tiles from 32 to 16 after one small-outer screen improved `1.86%` | Two fresh small-outer pairs improved `4.24%` and `0.44%`, while the production large-outer width-196 case regressed `25.43%` | A small launch-policy gain must repeat in every important outer regime. Use the project's retention gate, even when it is only 1%, but never average a major regime regression into a favorable aggregate. |
| Reduce very-short-row threads to 128/256/512, reduce rows per warp, or cap the grid at 32 | The tested candidates regressed `22.5%-259.9%` versus 1024 threads, two rows per warp, and grid 64 | Fewer threads, rows, or blocks can increase generated scheduling and fixed pipeline cost. Treat launch geometry as an orthogonal measured variable, not a monotonic resource-saving rule. |

## Measurement And Attribution Failures

| Failure pattern | Observed consequence | Required practice |
| --- | --- | --- |
| Treat separate stage files as repeated samples instead of stages of one call | Six-case published total appeared to regress `52.6%` when aggregation was corrected | Sum all stages per call before computing distributions across calls. |
| Let an operator-local path override an external package on `sys.path` | A 30-case run silently loaded the baseline binary and invalidated the candidate comparison | Resolve the loaded extension path and hash inside the benchmark process. |
| Reuse a run name and leave old profiler directories discoverable | Parser produced an invalid aggregate almost exactly twice the valid workflow | Use unique run names or clean profile trees; discard contaminated aggregates. |
| Change metric groups between latency comparisons | Detailed profiles may preserve attribution but perturb timing | Use a low-overhead timing run for retention and separate detailed runs for explanation. |
| Compare a component microbenchmark with a workflow-bound invocation using different materialized inputs | Absolute latency differed even though the kernel family looked similar | Normalize inputs, setup, and selected-kernel boundary before comparing. |
| Sum five selected kernel rows and call the result end-to-end launch cost | Host launch gaps and framework dispatch were omitted; the apparent distributed-Top-K margin was only a few microseconds under a plausible per-launch cost | Use selected-kernel sums for device-work attribution and a device event or synchronized wall interval for launch-reduction claims. Keep both boundaries. |
| Profile workflow components in separate processes while a downstream input binding re-executes its producer | The producer physically ran again inside the downstream raw profile, while name filtering excluded it and the published workflow synthetically added standalone component summaries | Audit physical launch sequence separately from aggregation. Prefer a one-process workflow runner that passes the exact device tensor and attributes both components from one execution. |
| Reuse a stale editable operator build during an otherwise clean profile pair | Candidate Sparse Attention appeared as `337.499 us`; an in-process module/device-ELF provenance hook exposed the wrong build and the run was discarded | Capture imported module, mapped device library, and hashes inside the benchmark process; a source root or preinstall log alone does not prove provenance. |
| Trust an idle DSMI frequency read or a correct-looking `PYTHONPATH` before a standard CannBench profile | DSMI reported rated 1650 MHz and idle 100 MHz while repeated accepted-format raw rows ran at `875/1650`; another run resolved a stale editable extension despite the intended source path | Gate each latency row on the raw profiler's execution-time current/rated frequency and on in-process extension path/hash. Profiler default warmup does not repair a node frequency state, and `PYTHONPATH` does not prove package provenance. |
| Increase exact-width-128 ownership from one to two rows per warp | Synchronized timing improved, but `BasicInfo` expanded the large-row cases to about `389 us`; the retained one-row form stayed near `24.4 us` on both boundaries | A source candidate can cross a profiler instrumentation cliff even when direct timing looks healthy. Validate on the publication boundary and retain the conflicting raw timing rather than averaging the boundaries together. |

## How To Apply These Results

1. Match phase, dtype, shape, hardware, compiler, and measurement boundary.
2. Use a result to prioritize or reject a hypothesis, not to predict an exact
   gain on another operator.
3. Re-run a rejected idea only when a named condition changed, such as layout,
   compiler, dominant boundary, available UB, or eliminated conversion cost.
4. Preserve negative results with the same rigor as winners: source/package
   provenance, accuracy, resource output, raw timings, and restoration proof.
5. For stacked changes, run the combined package. Interaction and unchanged
   stage variance make isolated percentages non-additive.

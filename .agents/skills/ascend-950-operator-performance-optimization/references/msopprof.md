# msOpProf Metric Selection And Artifacts

## Contents

- [Establish The Installed Contract](#establish-the-installed-contract)
- [Use The Exact Metric Grammar](#use-the-exact-metric-grammar)
- [Separate Machine And Human Artifacts](#separate-machine-and-human-artifacts)
- [Choose Standard CSV Metrics](#choose-standard-csv-metrics)
- [Choose Special And Control Metrics](#choose-special-and-control-metrics)
- [Disambiguate Similar Profiler Names](#disambiguate-similar-profiler-names)
- [Select Metrics From The Question](#select-metrics-from-the-question)
- [Collect In Separate Passes](#collect-in-separate-passes)
- [Apply Build Product And Replay Restrictions](#apply-build-product-and-replay-restrictions)
- [Use Auditable Commands](#use-auditable-commands)
- [Avoid Common Misreadings](#avoid-common-misreadings)

## Establish The Installed Contract

Treat the installed binary as the command-line authority and the matching
documentation as the semantic authority. Option names and support matrices can
change between CANN and MindStudio releases.

The observations in this reference were checked on the 20002 device node with:

```text
CANN:       9.2.0
msopprof:   26.2.0-9f53f86c867875b638157649a76e7d8ce8d32636
source tag: tag_MindStudio_26.2.0.B011_001
```

They were cross-checked against these current Chinese msOpProf references:

- [msOpProf user guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md)
  for options, features, and restrictions
- [msOpProf performance data](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/user_guide/msopprof_performance_data.md)
  for CSV field definitions

Before each campaign, capture the actual `msopprof --version` and
`msopprof --help` output. If the installed help disagrees with this reference,
use the installed spelling and reduce the collection to a small probe before
profiling the real workload. For example, the 26.2.0 help uses `PCSampling` and
`InstrTimeline`, while some documentation revisions spell them `PcSampling`
and `instrTimeLine`.

## Use The Exact Metric Grammar

The 26.2.0 binary accepts these 18 values for `--aic-metrics`:

```text
ArithmeticUtilization  MemoryUB              Memory
MemoryL0               L2Cache               PipeUtilization
ResourceConflictRatio  Default               BasicInfo
Roofline               Occupancy             TimelineDetail
KernelScale            Source                MemoryDetail
PCSampling             PipeTimeline          InstrTimeline
```

Join multiple values with an ASCII comma and no shell list syntax:

```bash
--aic-metrics=BasicInfo,Memory
```

Do not use `|`, spaces, or repeated assumptions about aliases. On the checked
binary, each of the following is invalid:

```text
--aic-metrics=BasicInfo|Memory
--aic-metrics=all-metrics
--aic-metrics=AllMetrics
--aic-metrics=all
--aic-metrics=ALL
```

There is no literal `all-metrics` value. Interpret an informal request for
"all metrics" before running anything:

- For all standard machine-readable PMU groups, use `Default`.
- Omitting `--aic-metrics` also selects `Default`.
- `Default` expands to exactly the seven standard groups:
  `ArithmeticUtilization`, `L2Cache`, `Memory`, `MemoryL0`, `MemoryUB`,
  `PipeUtilization`, and `ResourceConflictRatio`.
- `Default` does not enable every special visualization or instrumentation
  mode. Do not combine all 18 values blindly; some modes are incompatible,
  product-specific, or materially more intrusive.
- `KernelScale` by itself selects all seven standard groups, but only within
  code regions annotated by `MetricsProfStart` and `MetricsProfStop`. Add
  standard values after it to narrow the collection, for example
  `KernelScale,Memory,MemoryL0`.
- `Roofline` and `MemoryDetail` implicitly enable `Default`.
- `PCSampling` internally enables the source-related data needed for its
  visual analysis. This does not remove the debug-information requirement for
  source-line or call-stack mapping.

## Separate Machine And Human Artifacts

Classify the requested result before choosing a metric.

| Artifact | Primary consumer | Use it for |
| --- | --- | --- |
| `OpBasicInfo.csv` | scripts, dataframes, regression checks | launch identity, duration, block dimensions, device/process identity, and frequency gates |
| Seven metric CSV files | scripts, dataframes, automated diagnosis | numeric counter comparison, derived ratios, thresholds, and cross-run analysis |
| `visualize_data.bin` | MindStudio Insight | heatmaps, Roofline, occupancy, source/hotspot, warp-stall, and timeline views |
| `trace.json` | MindStudio Insight; Chrome tracing where supported | temporal pipeline inspection and event relationships |

The dedicated metric-to-CSV mapping contains only these eight files:

```text
OpBasicInfo.csv
ArithmeticUtilization.csv
L2Cache.csv
Memory.csv
MemoryL0.csv
MemoryUB.csv
PipeUtilization.csv
ResourceConflictRatio.csv
```

`BasicInfo` requests only `OpBasicInfo.csv`. The seven standard groups request
their correspondingly named CSV files; `Default` requests all seven. These are
the stable first choice for machine analysis.

The special modes do not have dedicated same-named CSV files. They primarily
add data to `visualize_data.bin` and, for timeline features, `trace.json`.
Treat these as visual-delivery modes intended for a person using MindStudio
Insight. A BIN-producing run can also contain standard CSV files when
`Default` is explicit or implied, so classify artifacts by file rather than
assuming that a run is exclusively "CSV" or "BIN."

`trace.json` is structured data, but its event schema is a visualization
contract rather than the standard metric-table contract. Do not make it the
default source for automated performance comparisons when an equivalent CSV
field exists.

## Choose Standard CSV Metrics

These groups are designed for numeric, machine-readable analysis. Exact
columns vary by product and software version, so inspect the generated header
and preserve it with the run.

| Metric and CSV | What it collects on Ascend 950 | Select it when |
| --- | --- | --- |
| `BasicInfo` -> `OpBasicInfo.csv` | operator name/type, task duration, block and mix-block dimensions, device/PID, current and rated frequency | establishing the canonical profiled latency, checking launch count/shape, or rejecting under-frequency samples |
| `ArithmeticUtilization` | Cube and Vector cycles, instruction counts and ratios; 950 fields distinguish Cube FP/INT and Vector VF/SFU/SIMT-VF activity | testing a compute-bound hypothesis, excess arithmetic, or poor Cube/Vector use |
| `L2Cache` | close/far L2 read/write hit, miss, victim counts, and hit rates on 950 | testing cache reuse, locality, cache imbalance, or excess GM traffic caused by L2 misses |
| `Memory` | GM, UB, L1, and L0C traffic, bandwidth/utilization, plus relevant MTE instruction counts and ratios | locating a broad data-movement bottleneck or deciding which memory level needs a focused pass |
| `MemoryL0` | L0A, L0B, and L0C read/write bandwidth | analyzing Cube feed/drain, L1-to-L0 movement, or L0C output pressure |
| `MemoryUB` | UB read/write bandwidth between Vector and GM-facing paths; 950 includes Vector/GM-to-UB and UB-to-Vector/GM views | analyzing Vector operand/result movement, UB pressure, or an expected GM-UB bottleneck |
| `PipeUtilization` | Cube, Vector, Scalar, FIXP, MTE1/2/3 time and ratios, I-cache misses, and active bandwidth fields | finding the dominant pipeline, insufficient compute/copy overlap, Scalar pressure, or a starved unit |
| `ResourceConflictRatio` | Cube/MTE waits and resource conflicts; 950 also exposes Vector STU/LDU/SFU conflict behavior | testing stalls from waits, bank/resource contention, or instruction scheduling conflicts |

Do not infer causality from one ratio. For example, high memory time can be a
symptom of insufficient arithmetic work, serialization, or replay effects.
Correlate CSV counters with workload size, source, launch geometry, and a
controlled change.

## Choose Special And Control Metrics

These values control scope or generate visual analysis rather than a dedicated
same-named CSV.

| Metric | Primary result and content | Use it when | Key restrictions |
| --- | --- | --- | --- |
| `Default` | all seven standard CSV groups and their data in the visual package | a broad machine-readable PMU sweep | not every special mode; higher overhead than `BasicInfo` |
| `Roofline` | `visualize_data.bin` Roofline views showing arithmetic intensity, achieved performance, hardware roofs, and compute- versus memory-bound advice; also implies `Default` CSVs | deciding whether the next optimization should reduce bytes or instructions | run separately when validating latency; invalid with 950 range replay |
| `Occupancy` | inter-core load view and CLI advice comparing duration, throughput, cache behavior, and 950 SIMT work where available | diagnosing core imbalance, skewed tiling, or uneven work distribution | supported on 950, A2, and A3; it is a physical-core comparison, not the conventional GPU occupancy percentage |
| `TimelineDetail` | simulated instruction timeline and source hotspot views derived alongside on-board data; add `Default` for standard CSV/complete heatmap data | PyTorch single-op API scenarios needing simulated instruction/source detail | no application or range replay; only supported call patterns; see restrictions below |
| `KernelScale` | scopes selected standard counters to annotated kernel regions | comparing phases inside one large kernel without attributing the whole kernel | requires paired `MetricsProfStart`/`MetricsProfStop` annotations; supported on 950/A2/A3 |
| `Source` | `visualize_data.bin` mapping from hot instructions/PCs to source and call stacks | locating hot code lines or navigating from cache/timeline evidence to source | compile with `-g` for the hotspot/source map; no MC2/LCCL; invalid with range replay |
| `MemoryDetail` | richer cache and data-path content in `visualize_data.bin`, including cache heatmaps and detailed MTE paths; also implies `Default` CSVs | inspecting L2-to-L0 connections, cacheline behavior, or detailed memory-path imbalance | supported on 950/A2/A3; invalid with range replay |
| `PCSampling` | `visualize_data.bin` SIMT warp-stall categories and source hotspot context | identifying why 950 SIMT warps stall: dependencies, barriers, resource/register conflicts, empty instruction buffer, and related causes | 950-only; raw sampling does not itself require `-g`, but source/call-stack mapping does; invalid with range replay |
| `PipeTimeline` | sampled per-pipe `trace.json`/`visualize_data.bin` timeline | inspecting coarse overlap, idle gaps, and active state across hardware pipes | 950-only; mutually exclusive with `InstrTimeline`; no communication-compute fused operators; at most six sampled cores |
| `InstrTimeline` | real on-board instruction timing in selected Cube, FIXP, Vector, or MTE pipes, delivered through `trace.json`/`visualize_data.bin` | instruction ordering, issue gaps, and scheduling analysis after a pipe is implicated | 950-only; mutually exclusive with `PipeTimeline`; no communication-compute fused operators; `-g` needed for PC/call stacks |

For `PipeTimeline`, dense `MarkStamp` points can lose data and SIMT functions do
not support `MarkStamp`. For `InstrTimeline`, each selected pipe is limited to
1024 instructions; inner SIMT VF and SIMD VF instructions are not displayed,
and dense instruction streams can lose data. Narrow the selected pipes and
reduce loop/instruction density in a diagnostic build when necessary.

## Disambiguate Similar Profiler Names

Do not collapse instruction-cache, instruction-buffer, and generic resource
stalls into one diagnosis:

- `aic_icache_miss_rate` / `aiv_icache_miss_rate` and
  `aic_scalar_wait_ib_time(us)` / `aiv_scalar_wait_ib_time(us)` are numeric
  `PipeUtilization.csv` fields. The wait field is Scalar IB wait for I-cache;
  correlate it with the matching I-cache miss rate before claiming an
  instruction-fetch bottleneck.
- `IBuf_Empty` is a SIMT warp-stall category from `PCSampling` in
  `visualize_data.bin`, not a `PipeUtilization.csv` column. The public 26.2.0
  PC-sampling categories are `IBuf_Empty`, `Nop_Cycles`,
  `Scoreboard_Not_Ready`, `Register_bank_conflict`, `Resource_conflict`,
  `Warp_Level_Sync`, `Divergence_Stack_Spill`, `Others`, and `Active`.
  The public guide renders the last name as lowercase `active`; preserve the
  spelling found in the actual artifact when parsing it.
- `dcache_entry_full` is neither an independent standard CSV field nor a
  public `PCSampling` category in the checked 26.2.0 contract. Do not relabel
  `Resource_conflict` as `dcache_entry_full`; retain any such lower-level
  explanation as a hypothesis until another documented counter or controlled
  experiment distinguishes it.

The timeline names also refer to different products:

| Name | Meaning in the checked 26.2.0 contract |
| --- | --- |
| `PipeUtilization` | numeric per-pipe utilization and wait fields in CSV |
| `PipeTimeline` | sampled on-board per-pipe timeline |
| `InstrTimeline` | real on-board instruction timing for selected pipes |
| `TimelineDetail` | simulated instruction timeline and source hotspot views |
| `PipeDetail` | not a valid `--aic-metrics` value |

Use the exact installed metric name and artifact instead of treating these
labels as aliases.

## Select Metrics From The Question

Use the smallest collection that can answer the current hypothesis:

| Question | First collection | Follow-up only if needed |
| --- | --- | --- |
| Which kernel ran, how long, how many times, and at what frequency? | `BasicInfo` | synchronized event/wall timing for end-to-end boundaries |
| I need all standard counters for automated analysis. | `Default` | one focused standard metric after the broad scan |
| Is it compute-bound or memory-bound? | `Roofline` | `ArithmeticUtilization`, `Memory`, then the implicated memory-level CSV |
| Which memory level or path is limiting? | `Memory` | `MemoryUB`, `MemoryL0`, `L2Cache`, or `MemoryDetail` according to the broad result |
| Are copy and compute pipelines overlapping? | `PipeUtilization` | `PipeTimeline`, then `InstrTimeline` for one or two implicated pipes |
| Are waits or resource conflicts dominating? | `ResourceConflictRatio,PipeUtilization` | `InstrTimeline` or `Source` |
| Are physical cores imbalanced? | `Occupancy` | inspect tiling ownership and compare per-core workload |
| Which source lines or instructions are hot? | `Source` with a `-g` build | `InstrTimeline` for ordering, or `PCSampling` for SIMT stalls |
| Why is a SIMT warp stalled? | `PCSampling` | `Source`/`InstrTimeline` with `-g` if source attribution is needed |
| Which region inside one kernel causes the counters? | `KernelScale,<focused-standard-metric>` | move or subdivide annotated regions in a new run |
| I need a person to inspect a complete visual report. | one hypothesis-specific visual metric, adding `Default` when required | import `visualize_data.bin` or `trace.json` into MindStudio Insight |

Do not collect `Default` merely because a human asked for "everything" if the
real question is a timeline or source hotspot. Conversely, do not select a
visual metric when the downstream consumer expects a stable CSV table.

## Collect In Separate Passes

Use a multi-pass workflow because detailed instrumentation perturbs execution
and some modes are incompatible:

1. Record the tool version, full command, application and device-library
   hashes, environment, target operator, shape, launch count, replay mode, and
   a unique output directory.
2. Collect `BasicInfo` for the canonical profiler latency, launch identity, and
   execution-time frequency gate.
3. Collect `Default` when broad machine-readable counters are needed.
4. State one bottleneck hypothesis and collect one focused standard or visual
   metric set.
5. Run `PipeTimeline`, `InstrTimeline`, source-related, and other intrusive or
   incompatible modes in separate output directories.
6. Inspect CSVs programmatically and import BIN/JSON artifacts into MindStudio
   Insight where visual correlation is the purpose.
7. Validate any optimization against a fresh `BasicInfo` and non-profiler
   baseline. Never use detailed-profile latency as the canonical speedup.

Preserve raw files. A parsed summary without the original CSV/BIN/JSON,
headers, version, and command is not auditable.

## Apply Build Product And Replay Restrictions

### General Restrictions

- Current msOpProf does not support operators compiled with `-O0`.
- Keep one profiling task per device. The guide recommends a collection under
  five minutes and more than 20 GB of available/configured memory.
- Use a current-user output path without symbolic-link components and a unique
  output root for every run.
- The profiler can instrument or modify an input shared library in place in
  some workflows. Profile a work copy, retain the untouched build, and compare
  hashes before and after profiling. See
  [compiler-and-runtime-pitfalls.md](compiler-and-runtime-pitfalls.md).
- A `-g` build contains debug information. Restrict access to it and do not
  silently substitute it for the production-timing artifact.

### Debug Information

Compile with `-g` when the requested result needs:

- the `Source` hotspot/source map
- instruction PC and call-stack display in `InstrTimeline`
- source or call-stack navigation from cache or timeline views

Do not claim that raw `PCSampling` requires `-g`; the documented requirement is
for mapping sampled PCs to source/call stacks. Without useful debug mapping,
retain the raw stall categories but state that source attribution is missing.

### Replay Modes

- `kernel` is the default replay mode.
- `kernel` and `range` replay clear L2 cache; `application` replay does not.
  Do not compare cache metrics or latency across these modes as equivalent.
- `TimelineDetail` is incompatible with `application` and `range` replay.
- `range` requires `--mstx=on`, paired non-crossing range markers, unchanged
  streams within a range, warmup of at least one, and preferably no more than
  50 operators in one range.
- Range replay does not support MC2 or LCCL operators.
- On Ascend 950, range replay is incompatible with `PCSampling`, `Source`,
  `Roofline`, `PipeTimeline`, and `InstrTimeline`; also avoid `--kill=on`.
  `TimelineDetail` and `MemoryDetail` are independently documented as
  range-incompatible.
- `application` replay does not support multi-device multi-operator scenarios.
  Some special-metric application runs can produce incomplete
  `visualize_data.bin`; add `Default` when a complete visual package is needed.

### Metric And Operator Support

- `PipeTimeline`, `InstrTimeline`, and `PCSampling` are Ascend 950-only in the
  checked release.
- `PipeTimeline` and `InstrTimeline` cannot be enabled together.
- `PipeTimeline` and `InstrTimeline` do not support communication-compute fused
  operators.
- `TimelineDetail` supports only third-party PyTorch operator calls that invoke
  the operator through a single-op API. It does not support secondary-pointer
  operators, Triton operators, or communication-compute fused operators.
  Although the metric supports 950, its `--dump` and `--core-id` helper options
  are documented only for A2/A3 and do not take effect on 950.
- `Source` hotspot maps do not support MC2 or LCCL operators.
- Product, operator-type, and view support can be narrower than parser
  acceptance. Check the installed guide and verify that the expected artifact
  is non-empty before relying on it.

## Use Auditable Commands

Use the command front end installed with the target CANN release. Some
environments expose `msopprof` directly and others document `msprof op`; keep
the installed invocation form in the run record. Representative direct-binary
commands are:

```bash
# Canonical profiler timing/frequency/launch baseline.
msopprof --aic-metrics=BasicInfo --launch-count=10 \
  --output=<basic-info-dir> <app> [args]

# All seven standard machine-readable metric CSVs.
msopprof --aic-metrics=Default \
  --output=<default-dir> <app> [args]

# Focused CSV collection.
msopprof --aic-metrics=Memory,MemoryUB,PipeUtilization \
  --output=<memory-dir> <app> [args]

# Annotated kernel region with only selected standard groups.
msopprof --aic-metrics=KernelScale,Memory,MemoryL0 \
  --output=<region-dir> <app> [args]

# Human-oriented visual investigations.
msopprof --aic-metrics=Roofline \
  --output=<roofline-dir> <app> [args]
msopprof --aic-metrics=Source \
  --output=<source-dir> <g-built-app> [args]
msopprof --aic-metrics=PCSampling \
  --output=<pc-sampling-dir> <app> [args]
msopprof --aic-metrics=PipeTimeline \
  --output=<pipe-timeline-dir> <app> [args]
msopprof --aic-metrics=InstrTimeline \
  --instr-timeline-pipe="mte1|vector" \
  --output=<instr-timeline-dir> <g-built-app> [args]
```

The comma separates metrics inside `--aic-metrics`; the vertical bar is valid
inside `--instr-timeline-pipe` to select multiple pipes. Do not transfer one
option's separator grammar to another.

## Avoid Common Misreadings

- `Default` means all seven standard PMU CSV groups, not every mode supported
  by the binary.
- `BasicInfo` is not part of the seven `Default` metric names even though an
  msOpProf output package can include operator basic information.
- A metric accepted by the parser is not proof that the current product,
  operator type, replay mode, or build supports the requested view.
- A BIN-only special metric has no dedicated same-named CSV. The run may still
  contain CSVs because `Default` was added or implied.
- `visualize_data.bin` is for MindStudio Insight. Do not parse it as an ad hoc
  binary format when the supported viewer or a CSV answers the question.
- `trace.json` is not interchangeable with standard metric CSVs even though it
  is text and can be parsed.
- Profiler replay and instrumentation can change caches, frequency, binary
  contents, and timing. Use them to explain a bottleneck, then retest the
  candidate on the frozen performance boundary.

from pathlib import Path


SIMT_OP_ROOT = Path(
    "src/cannbench/operators/builtin/softmax/simt/v1/"
    "aten_softmax/csrc/simt"
)
SIMT_OP_V2_ROOT = Path(
    "src/cannbench/operators/builtin/softmax/simt/v2/"
)
SIMT_OP_V3_ROOT = Path(
    "src/cannbench/operators/builtin/softmax/simt/v3/"
)
SIMT_OP_V3_SIMT_ROOT = SIMT_OP_V3_ROOT / "aten_softmax_v3" / "csrc" / "simt"


def _read_v3_simt_source(name: str) -> str:
    return (SIMT_OP_V3_SIMT_ROOT / name).read_text()


def _read_v3_simt_sources(*names: str) -> str:
    return "\n".join(_read_v3_simt_source(name) for name in names)


def test_ascend_softmax_keeps_dim_parallel_launch_policy():
    header = (SIMT_OP_ROOT / "occupancy_common.h").read_text()

    assert "dim_threads *= 2" in header
    assert "return {dim_threads, inner_threads};" in header


def test_ascend_softmax_reduction_uses_dynamic_ubuf_not_global_scratch():
    source = (SIMT_OP_ROOT / "spatial_softmax.asc").read_text()

    assert "extern __ubuf__ uint32_t dynamicStartUB[];" in source
    assert "(__ubuf__ accscalar_t*)dynamicStartUB" in source
    assert "spatial_block_reduce_x" in source
    assert "launch.dynamic_ubuf_bytes" in source
    assert "accscalar_t* reduce_workspace" not in source
    assert "at::empty(\n      {\n          launch.grid_x * launch.grid_y" not in source


def test_ascend_softmax_v2_uses_versioned_python_package_and_torch_namespace():
    setup_py = (SIMT_OP_V2_ROOT / "setup.py").read_text()
    ops_py = (SIMT_OP_V2_ROOT / "aten_softmax_v2" / "ops.py").read_text()
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    header = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "occupancy_common.h"
    ).read_text()

    assert 'library_name = "aten_softmax_v2"' in setup_py
    assert "torch.ops.aten_softmax_v2" in ops_py
    assert "namespace aten_softmax_v2" in source
    assert "namespace aten_softmax_v2::simt" in header
    assert "aten_softmax_v2_v2" not in source
    assert "aten_softmax_v2_v2" not in header
    assert "TORCH_LIBRARY_FRAGMENT(aten_softmax_v2, m)" in source
    assert "TORCH_LIBRARY_IMPL(aten_softmax_v2, PrivateUse1, m)" in source


def test_ascend_softmax_v2_splits_rowwise_dispatch_like_cuda():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "enum class RowSoftmaxPath" in source
    assert "PersistentLike" in source
    assert "FastLike" in source
    assert "GenericLike" in source
    assert "dim_size <= 2048 && dim_size * scalar_size <= 8192" in source
    assert "aten_softmax_v2::row_softmax_persistent_forward" in source
    assert "aten_softmax_v2::row_softmax_fast_forward" in source
    assert "aten_softmax_v2::row_softmax_generic_forward" in source


def test_ascend_softmax_v2_rowwise_launch_policy_is_not_fixed_to_32_threads():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_persistent_block_x" in source
    assert "row_softmax_fast_block_x" in source
    assert "row_softmax_generic_block_x" in source
    assert "constexpr int64_t kCudaFastPathThreads = 512" in source
    assert "constexpr int64_t kCudaGenericMaxThreads = 1024" in source


def test_ascend_softmax_v2_persistent_path_uses_multi_row_block_shape():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    persistent_start = source.index("row_softmax_persistent_forward_kernel")
    fast_start = source.index("row_softmax_fast_forward_kernel")
    persistent_source = source[persistent_start:fast_start]

    assert "row_softmax_persistent_forward_kernel" in source
    assert "row_softmax_persistent_block_y" in source
    assert "threadIdx.y" in persistent_source
    assert "dim3(block_x, block_y)" in source
    assert "blockDim.y * blockIdx.x + threadIdx.y" in persistent_source


def test_ascend_softmax_v3_persistent_path_uses_shape_aware_threads_per_block():
    source = _read_v3_simt_source("spatial_softmax.asc")

    assert "inline int64_t row_softmax_persistent_threads_per_block(int64_t dim_size)" in source
    assert "if (dim_size == 512) {\n    return 1024;" in source
    assert "if (dim_size == 1024) {\n    return 256;" in source
    assert "return 1024;" in source
    assert "row_softmax_persistent_block_y(block_x, dim_size)" in source


def test_ascend_softmax_v3_persistent_kernel_uses_shape_aware_launch_bounds():
    source = _read_v3_simt_source("row_persistent_fallback.asc")
    vf_template = source[
        source.rindex("int64_t kThreadsPerBlock", 0, source.index("row_softmax_persistent_forward_vf"))
        : source.index("row_softmax_persistent_forward_kernel")
    ]
    kernel_template = source[
        source.rindex("int64_t kThreadsPerBlock", 0, source.index("row_softmax_persistent_forward_kernel"))
        : source.index("launch_row_persistent_forward_kernel")
    ]

    assert "__launch_bounds__(kThreadsPerBlock)" in vf_template
    assert "int64_t kThreadsPerBlock" in vf_template
    assert "int64_t kThreadsPerBlock" in kernel_template
    assert "row_softmax_persistent_forward_kernel" in kernel_template


def test_ascend_softmax_v3_isolates_256_512_and_1024_persistent_kernels():
    source = _read_v3_simt_source("row_persistent_fallback.asc")
    bucket_common = _read_v3_simt_source("persistent_bucket_pad_common.h")
    persistent_128 = _read_v3_simt_source("persistent_128.asc")
    persistent_256 = _read_v3_simt_source("persistent_256.asc")
    persistent_512 = _read_v3_simt_source("persistent_512.asc")
    persistent_1024 = _read_v3_simt_source("persistent_1024.asc")

    assert "void dispatch_row_persistent_forward_kernel_128_fp16(" in source
    assert "void dispatch_row_persistent_forward_kernel_128_fp32(" in source
    assert "void dispatch_row_persistent_forward_kernel_256_fp16(" in source
    assert "void dispatch_row_persistent_forward_kernel_256_fp32(" in source
    assert "void dispatch_row_persistent_forward_kernel_512_fp16(" in source
    assert "void dispatch_row_persistent_forward_kernel_512_fp32(" in source
    assert "void dispatch_row_persistent_forward_kernel_1024_fp16(" in source
    assert "void dispatch_row_persistent_forward_kernel_1024_fp32(" in source
    assert 'TORCH_CHECK(false, "unsupported <=128 persistent dtype combination");' in source
    assert 'TORCH_CHECK(false, "unsupported 257-512 persistent dtype combination");' in source
    assert 'TORCH_CHECK(false, "unsupported 1024-thread persistent dtype combination");' in source
    assert 'TORCH_CHECK(false, "unsupported 129-256 persistent dtype combination");' in source
    assert '#include "persistent_bucket_pad_common.h"' in persistent_128
    assert '#include "persistent_bucket_pad_common.h"' in persistent_256
    assert '#include "persistent_bucket_pad_common.h"' in persistent_512
    assert '#include "c_api/asc_simd.h"' in bucket_common
    assert "__launch_bounds__(kPersistentBucketThreadsPerBlock)" in bucket_common
    assert "64>" in persistent_128
    assert "128>" in persistent_128
    assert "256>" in persistent_256
    assert "512>" in persistent_512
    assert "row_offset = row_in_tile * kBucketCols" in bucket_common
    assert "row_softmax_persistent_bucket_pad_kernel" in bucket_common
    assert "asc_copy_gm2ub_align" in bucket_common
    assert "asc_copy_ub2gm_align" in bucket_common
    assert "padded_input_row_bytes" in bucket_common
    assert "padded_output_row_bytes" in bucket_common
    assert "input_right_padding_elems" in bucket_common
    assert "asc_reduce_max(max_value)" in bucket_common
    assert "asc_reduce_add(sum_value)" in bucket_common
    assert "warp_reduce<" not in bucket_common
    assert "asc_shfl_xor" not in bucket_common
    assert "dispatch_row_persistent_forward_kernel_with_512_threads" in persistent_512
    assert '#include "c_api/asc_simd.h"' in persistent_1024
    assert "__launch_bounds__(256)" in persistent_1024
    assert "dispatch_row_persistent_forward_kernel_with_1024_threads" in persistent_1024
    assert "row_softmax_persistent_hybrid_pipeline_kernel" in persistent_1024
    assert "asc_copy_gm2ub_align" in persistent_1024
    assert "asc_copy_ub2gm_align" in persistent_1024
    assert "asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);" in persistent_1024
    assert "asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID1);" in persistent_1024


def test_ascend_softmax_v3_bucket_persistent_paths_use_padded_strided_copy():
    bucket_common = _read_v3_simt_source("persistent_bucket_pad_common.h")

    copy_in = bucket_common.index("asc_copy_gm2ub_align(")
    current_vf = bucket_common.index("asc_vf_call<row_softmax_persistent_bucket_pad_vf<")
    copy_out = bucket_common.index("asc_copy_ub2gm_align(")

    assert copy_in < current_vf < copy_out
    assert "input_tile_ub\n      [2][kPersistentBucketRowsPerTile * kBucketCols]" in bucket_common
    assert "output_tile_ub\n      [2][kPersistentBucketRowsPerTile * kBucketCols]" in bucket_common
    assert "static_cast<uint16_t>(row_count)" in bucket_common
    assert "row_bytes,\n        padded_input_row_bytes" in bucket_common
    assert "output_row_bytes,\n        padded_output_row_bytes" in bucket_common
    assert "asc_sync_wait(PIPE_MTE3, PIPE_V, event_id)" in bucket_common
    assert "basic_api/" not in bucket_common
    assert "kernel_operator.h" not in bucket_common


def test_ascend_softmax_v3_1024_persistent_path_prefetches_next_tile():
    persistent_1024 = _read_v3_simt_source("persistent_1024.asc")

    next_tile = persistent_1024.index("if (tile_idx + 1 < tile_count)")
    prefetch = persistent_1024.index("asc_copy_gm2ub_align(", next_tile)
    current_vf = persistent_1024.index(
        "asc_vf_call<row_softmax_persistent_hybrid_vf<", next_tile
    )
    assert prefetch < current_vf
    assert "input_tile_ub[2]" in persistent_1024
    assert "output_tile_ub[2]" in persistent_1024
    assert "asc_sync_wait(PIPE_V, PIPE_MTE2, next_event_id)" in persistent_1024
    assert "asc_sync_notify(PIPE_MTE2, PIPE_V, next_event_id)" in persistent_1024
    assert "asc_sync_wait(PIPE_MTE3, PIPE_V, event_id)" in persistent_1024


def test_ascend_softmax_v3_dispatches_le_128_to_bucket_pad_pipeline():
    dispatch = _read_v3_simt_source("row_persistent_fallback.asc")
    persistent_128 = _read_v3_simt_source("persistent_128.asc")
    le_128_start = dispatch.index("if (dim_size <= 128)")
    le_128_end = dispatch.index(
        "if (dim_size >= 129 && dim_size <= 256)", le_128_start
    )
    le_128_dispatch = dispatch[le_128_start:le_128_end]

    assert "if (dim_size <= 128)" in dispatch
    assert "dispatch_row_persistent_forward_kernel_128_fp16(" in le_128_dispatch
    assert "dispatch_row_persistent_forward_kernel_128_fp32(" in le_128_dispatch
    assert '#include "persistent_bucket_pad_common.h"' in persistent_128
    assert "dim_size <= 64" in persistent_128
    assert "64>" in persistent_128
    assert "128>" in persistent_128
    assert "basic_api/" not in persistent_128
    assert "kernel_operator.h" not in persistent_128


def test_ascend_softmax_v3_dispatches_129_through_256_to_ub_pipeline():
    source = _read_v3_simt_source("row_persistent_fallback.asc")

    assert "dim_size >= 129 && dim_size <= 256" in source
    assert "dispatch_row_persistent_forward_kernel_256_fp16(" in source
    assert "dispatch_row_persistent_forward_kernel_256_fp32(" in source


def test_ascend_softmax_v3_remaining_persistent_fallback_uses_generic_1024_thread_kernel():
    source = _read_v3_simt_source("row_persistent_fallback.asc")

    assert "if (dim_size == 1024) {" in source
    assert "if (dim_size >= 257 && dim_size <= 512) {" in source
    assert "dispatch_row_persistent_forward_kernel_with_threads<" in source
    generic_section = source[source.rindex("dispatch_row_persistent_forward_kernel_with_threads<") :]
    assert "1024>(" in generic_section
    assert "dispatch_row_persistent_forward_kernel_1024_fp16(" not in generic_section
    assert "dispatch_row_persistent_forward_kernel_1024_fp32(" not in generic_section


def test_ascend_softmax_v2_persistent_path_uses_cuda_style_register_warp_kernel():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    persistent_start = source.index("row_softmax_persistent_forward_kernel")
    fast_start = source.index("row_softmax_fast_forward_kernel")
    persistent_source = source[persistent_start:fast_start]

    assert "persistent_warp_reduce" in source
    assert "asc_shfl_xor" in source
    assert "constexpr int64_t kWarpIterations" in persistent_source
    assert "constexpr int64_t kWarpBatch" in persistent_source
    assert "accscalar_t elements[kWarpBatch][kWarpIterations]" in persistent_source
    assert "row_base = (blockDim.y * blockIdx.x + threadIdx.y) * kWarpBatch" in persistent_source
    assert "persistent_warp_reduce<accscalar_t, kWarpBatch, kWarpSize, Max>" in persistent_source
    assert "spatial_block_reduce_x" not in persistent_source


def test_ascend_softmax_v2_fast_path_has_dedicated_multi_row_kernel():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "row_softmax_fast_forward_kernel" in source
    assert "path == RowSoftmaxPath::FastLike" in source
    assert "path == RowSoftmaxPath::FastLike\n      ? 1" in source
    assert "aten_softmax_v2::row_softmax_fast_forward" in source


def test_ascend_softmax_v2_fast_path_uses_cuda_style_block_reduce_kernel():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    fast_start = source.index("row_softmax_fast_forward_kernel")
    generic_start = source.index("row_softmax_generic_forward_kernel")
    fast_source = source[fast_start:generic_start]

    assert "fast_ilp_reduce" in source
    assert "fast_block_reduce_warp" in source
    assert "fast_block_reduce_warp_inverse" in source
    assert "constexpr int64_t kFastILP" in fast_source
    assert "row_input = input + row * dim_size" in fast_source
    assert "row_output = output + row * dim_size" in fast_source
    assert "for (int64_t offset = threadIdx.x; offset < dim_size;" in fast_source
    assert "offset += blockDim.x" in fast_source
    assert "row = blockIdx.x * blockDim.y + threadIdx.y" not in fast_source
    assert "row_softmax_fast_block_x() {\n  constexpr int64_t kCudaFastPathThreads = 512;" in source
    assert "row_softmax_fast_ubuf_bytes" in source
    assert "row_softmax_fast_block_x() / kCudaWarpLaneLimit" in source


def test_ascend_softmax_v3_fast_path_uses_1024_threads_per_block():
    source = _read_v3_simt_source("spatial_softmax.asc")

    assert "row_softmax_fast_block_x() {\n  constexpr int64_t kCudaFastPathThreads = 1024;" in source


def test_ascend_softmax_v3_fast_path_uses_real_ilp_and_shifted_half2():
    source = _read_v3_simt_source("row_fast.asc")
    fast_start = source.index("row_softmax_fast_forward_vf")
    reg_start = source.index("row_softmax_fast_reg_forward_vf")
    fast_source = source[fast_start:reg_start]

    assert "constexpr int64_t kFastILP = 4;" in source
    assert "offset = threadIdx.x * kILP" in source
    assert "offset += blockDim.x * kILP" in source
    assert "fast_ilp_softmax_write" in source
    assert "fast_fp16x2_reduce" in source
    assert "fast_fp16x2_write" in source
    assert "const int64_t fp16x2_shift = row_softmax_fast_fp16x2_shift(row, dim_size);" in source
    assert "aligned_data = data + shift" in source
    assert "aligned_input = input + shift" in source


def test_ascend_softmax_v3_fast_path_has_register_cache_variant():
    source = _read_v3_simt_source("row_fast.asc")

    assert "row_softmax_fast_reg_forward_vf" in source
    assert "row_softmax_fast_reg_forward_kernel" in source
    assert "row_softmax_fast_register_count" in source
    assert "use_register_like_row_softmax_fast" in source
    assert "register_count > 1 && register_count <= 8" in source
    assert "scalar_t elements[kRegCount]" in source
    assert "dispatch_row_fast_forward_kernel" in source
    assert "launch_row_fast_reg_forward_kernel" in source


def test_ascend_softmax_v3_fast_path_uses_large_row_recompute_for_logits_scale_dims():
    source = _read_v3_simt_source("row_fast.asc")

    assert "inline bool use_large_row_recompute_row_softmax_fast(int64_t dim_size)" in source
    assert "constexpr int64_t kUsableUbBytes = 224 * 1024;" in source
    assert "constexpr int64_t kLargeRowUbDataBytes = 220 * 1024;" in source
    assert "kLargeRowUbDataBytes / (2 * sizeof(__fp16))" in source
    assert "kLargeRowUbDataBytes / (2 * sizeof(float))" in source
    assert "? kLargeRowFp16TileElements\n      : kLargeRowFp32TileElements" in source
    assert "return dim_size >= 8192 && dim_size <= kMaxWholeRowElements;" in source
    assert "row_softmax_fast_large_row_recompute_impl" in source
    assert "row_softmax_fast_large_row_recompute_kernel" in source
    assert "launch_row_fast_large_recompute_forward_kernel" in source
    assert "row_softmax_fast_large_row_inplace_pipeline_impl" in source
    assert "row_softmax_fast_large_row_inplace_pipeline_kernel" in source
    assert "launch_row_fast_large_inplace_pipeline_forward_kernel" in source
    assert "row_softmax_fast_large_row_forward_vf" in source
    assert "asc_copy_gm2ub_align_sync" in source
    assert "asc_copy_ub2gm_align_sync" in source
    pipeline_start = source.index("row_softmax_fast_large_row_inplace_pipeline_impl")
    pipeline_end = source.index(
        "row_softmax_fast_large_row_inplace_pipeline_kernel", pipeline_start
    )
    pipeline_source = source[pipeline_start:pipeline_end]
    assert "row_tile_ub[2][kTileElements]" in pipeline_source
    assert "asc_copy_gm2ub_align(" in pipeline_source
    assert "asc_copy_ub2gm_align(" in pipeline_source
    assert "asc_copy_gm2ub_align_sync" not in pipeline_source
    assert "asc_copy_ub2gm_align_sync" not in pipeline_source
    assert "row_tile_ub[slot],\n        row_tile_ub[slot]," in pipeline_source
    assert "asc_sync_notify(PIPE_MTE2, PIPE_V, next_event_id)" in pipeline_source
    assert "asc_sync_notify(PIPE_V, PIPE_MTE3, event_id)" in pipeline_source
    next_row_start = pipeline_source.index("if (row_idx + 1 < row_count)")
    prefetched_copy = pipeline_source.index("asc_copy_gm2ub_align(", next_row_start)
    current_row_compute = pipeline_source.index(
        "asc_vf_call<row_softmax_fast_large_row_forward_vf<", next_row_start
    )
    assert prefetched_copy < current_row_compute
    assert "const accscalar_t row_max = fast_block_reduce_warp<accscalar_t, Max>" in source
    assert "const accscalar_t inv_sum = fast_block_reduce_warp_inverse<accscalar_t, Add>" in source
    assert "dim3(kThreadsPerBlock)," in source
    assert "kPipelinePhysicalGridXLimit = 64" in source
    assert "<<<pipeline_grid_x, 0, acl_stream>>>" in source
    assert "32768>(" in source
    assert "kLargeRowFp16TileElements>(" in source
    assert "kLargeRowFp32TileElements>(" in source


def test_ascend_softmax_v3_large_rows_use_mte_tiled_pipeline_with_fp16_fused_no_workspace():
    row_fast_source = _read_v3_simt_source("row_fast.asc")
    spatial_source = _read_v3_simt_source("spatial_softmax.asc")

    assert "launch_row_fast_large_tiled_forward_kernel" in row_fast_source
    assert "dispatch_row_fast_large_tiled_forward_kernel" in row_fast_source
    assert "dispatch_row_fast_large_tiled_fused_forward_kernel" in row_fast_source
    assert "dispatch_row_fast_large_tiled_fused_forward_kernel" in spatial_source
    assert "row_softmax_fast_large_ub_tiled_stats_pipeline_kernel<" in row_fast_source
    assert "row_softmax_fast_large_ub_tiled_write_pipeline_kernel<" in row_fast_source
    assert "row_softmax_fast_large_ub_tiled_fused_pipeline_kernel<" in row_fast_source
    assert "row_softmax_fast_large_ub_tiled_local_stats_vf" in row_fast_source
    assert "row_softmax_fast_large_ub_tiled_write_local_stats_vf" in row_fast_source
    fused_start = row_fast_source.index(
        "row_softmax_fast_large_ub_tiled_fused_pipeline_impl"
    )
    fused_end = row_fast_source.index(
        "row_softmax_fast_large_ub_tiled_fused_pipeline_kernel", fused_start
    )
    fused_source = row_fast_source[fused_start:fused_end]
    assert "const int32_t retained_tile_idx = tile_count - 2;" in fused_source
    assert "const int32_t retained_slot = (tile_count - 1) & 1;" in fused_source
    assert "row_softmax_fast_large_ub_tiled_write_local_stats_vf<scalar_t>" in fused_source
    assert "const int32_t remaining_write_count = tile_count - 1;" in fused_source
    assert "row_softmax_fast_large_gmem_max_vf" not in row_fast_source
    assert "row_softmax_fast_large_gmem_sum_vf" not in row_fast_source
    assert "row_softmax_fast_large_gmem_write_vf" not in row_fast_source
    assert "row_softmax_fast_large_gmem_max_kernel" not in row_fast_source
    assert "row_softmax_fast_large_gmem_sum_kernel" not in row_fast_source
    assert "row_softmax_fast_large_gmem_write_kernel" not in row_fast_source
    assert "use_large_row_workspace_row_softmax_fast" in spatial_source
    assert "kMaxWholeRowElements = kFp16Path ? 56320 : 28160" in spatial_source
    assert "row_max_ptr == nullptr" in spatial_source
    assert "use_large_row_workspace_row_softmax_fast<scalar_t, outscalar_t>(dim_size)" in spatial_source
    half_branch_start = spatial_source.index(
        "if (in_tensor.scalar_type() == at::ScalarType::Half)"
    )
    half_branch_end = spatial_source.index(
        "} else if (in_tensor.scalar_type() == at::ScalarType::Float)",
        half_branch_start,
    )
    half_branch = spatial_source[half_branch_start:half_branch_end]
    assert "auto row_max = at::empty({outer_size}" not in half_branch
    assert "auto row_inv_sum = at::empty({outer_size}" not in half_branch
    assert "auto row_max = at::empty({outer_size}" in spatial_source
    assert "auto row_inv_sum = at::empty({outer_size}" in spatial_source
    assert "row_max.mutable_data_ptr<float>()" in spatial_source
    assert "row_inv_sum.mutable_data_ptr<float>()" in spatial_source


def test_ascend_softmax_v3_fp16_50k_rows_use_single_inplace_ub_pipeline():
    row_fast_source = _read_v3_simt_source("row_fast.asc")
    spatial_source = _read_v3_simt_source("spatial_softmax.asc")

    assert "kLargeRowFp16TileElements" in row_fast_source
    assert "kMaxWholeRowElements = kFp16Path ? 56320 : 28160" in spatial_source
    assert "launch_row_fast_large_inplace_pipeline_forward_kernel<" in row_fast_source
    assert "kLargeRowFp16TileElements>(" in row_fast_source
    assert "if (dim_size <= 32768)" in row_fast_source
    fused_vf_start = row_fast_source.index("row_softmax_fast_large_row_forward_vf")
    fused_vf_end = row_fast_source.index(
        "row_softmax_fast_large_row_recompute_impl", fused_vf_start
    )
    fused_vf_source = row_fast_source[fused_vf_start:fused_vf_end]
    assert "reinterpret_cast<__ubuf__ const half2*>(input_ub)" in fused_vf_source
    assert "reinterpret_cast<__ubuf__ half2*>(output_ub)" in fused_vf_source
    assert "__floats2half2_rn(" in fused_vf_source
    assert "row_softmax_fast_large_ub_stats_vf" not in row_fast_source
    assert "row_softmax_fast_large_ub_stats_pipeline_kernel" not in row_fast_source


def test_ascend_softmax_v3_large_fp16_ub_reductions_use_half2_with_tail_fallback():
    source = _read_v3_simt_source("row_fast.asc")
    whole_row_start = source.index("row_softmax_fast_large_row_forward_vf")
    whole_row_end = source.index(
        "row_softmax_fast_large_row_recompute_impl", whole_row_start
    )
    tiled_start = source.index(
        "row_softmax_fast_large_ub_tiled_update_running_stats"
    )
    tiled_end = source.index("row_softmax_fast_large_ub_tiled_stats_vf", tiled_start)

    for section, input_name, size_name in (
        (source[whole_row_start:whole_row_end], "input_ub", "tile_size"),
        (source[tiled_start:tiled_end], "tile_ub", "tile_elements"),
    ):
        half2_cast = (
            "reinterpret_cast<__ubuf__ const half2*>(" + input_name + ")"
        )
        assert section.count(half2_cast) >= 2
        assert section.count("__low2float(") >= 4
        assert section.count("__high2float(") >= 4
        assert section.count("quad_half2_size = half2_size & ~3") >= 2
        assert section.count("pair_offset < quad_half2_size") >= 2
        assert "if (pair_offset + 1 < half2_size)" not in section
        assert section.count("tail_pair =") >= 2
        assert section.count(f"if (({size_name} % 2) != 0 && threadIdx.x == 0)") >= 2
        assert "offset = threadIdx.x * kLargeRowILP" in section
        assert "offset += blockDim.x * kLargeRowILP" in section


def test_ascend_softmax_v3_fp16_and_fp32_huge_rows_use_tiled_online_stats():
    source = _read_v3_simt_source("row_fast.asc")

    assert "row_softmax_fast_large_ub_tiled_stats_vf" in source
    assert "template <typename scalar_t, int64_t kTileElements>" in source
    assert "row_softmax_fast_large_ub_tiled_stats_pipeline_kernel<" in source
    assert "scalar_t,\n      kTileElements>" in source
    assert "__ubuf__ float running_stats_ub[2];" in source
    assert "previous_sum * __expf(previous_max - combined_max)" in source
    assert "tile_sum * __expf(tile_max - combined_max)" in source
    assert "next_tile_elements * sizeof(scalar_t)" in source
    assert "row_inv_sum[row] = 1.0f / running_stats[1];" in source


def test_ascend_softmax_v3_large_rows_pipeline_tiled_final_write():
    source = _read_v3_simt_source("row_fast.asc")

    assert "row_softmax_fast_large_ub_tiled_write_vf" in source
    assert "row_softmax_fast_large_ub_tiled_write_pipeline_kernel<" in source
    assert "kPhysicalGridXLimit = 64" in source

    pipeline_start = source.index(
        "row_softmax_fast_large_ub_tiled_write_pipeline_impl"
    )
    pipeline_end = source.index(
        "row_softmax_fast_large_ub_tiled_write_pipeline_kernel", pipeline_start
    )
    pipeline_source = source[pipeline_start:pipeline_end]
    assert "__ubuf__ scalar_t row_tile_ub[2][kTileElements];" in pipeline_source
    assert "asc_copy_gm2ub_align(" in pipeline_source
    assert "asc_copy_ub2gm_align(" in pipeline_source
    assert "next_tile_elements * sizeof(scalar_t)" in pipeline_source
    assert "tile_elements * sizeof(scalar_t)" in pipeline_source
    assert "asc_sync_wait(PIPE_MTE3, PIPE_V, next_event_id)" in pipeline_source
    assert "asc_sync_notify(PIPE_MTE2, PIPE_V, next_event_id)" in pipeline_source
    assert "asc_sync_notify(PIPE_V, PIPE_MTE3, event_id)" in pipeline_source
    assert "asc_sync_notify(PIPE_MTE3, PIPE_V, event_id)" in pipeline_source
    next_tile_start = pipeline_source.index("const bool has_next_tile")
    prefetched_copy = pipeline_source.index(
        "asc_copy_gm2ub_align(", next_tile_start
    )
    current_tile_compute = pipeline_source.index(
        "asc_vf_call<row_softmax_fast_large_ub_tiled_write_vf<scalar_t>>",
        next_tile_start,
    )
    assert prefetched_copy < current_tile_compute


def test_ascend_softmax_v3_build_releases_reserved_ub_regions():
    setup_source = (SIMT_OP_V3_ROOT / "setup.py").read_text()

    assert '"--cce-disable-vf-stack-reserved-ubuf"' in setup_source
    assert '"--cce-disable-asc-reserved-ubuf"' in setup_source


def test_ascend_softmax_v3_uses_mixed_simd_simt_vf_launch_model():
    source = _read_v3_simt_sources(
        "row_persistent_fallback.asc",
        "row_fast.asc",
        "spatial_kernel.asc",
    )

    assert "__launch_bounds__(kThreadsPerBlock)" in source
    assert "row_softmax_persistent_forward_vf(" in source
    assert "__simt_vf__ __launch_bounds__(1024) inline void row_softmax_fast_forward_vf" in source
    assert "__simt_vf__ __launch_bounds__(1024) inline void row_softmax_fast_reg_forward_vf" in source
    assert "__simt_vf__ inline void cunn_spatial_softmax_forward_vf" in source
    assert "__global__ __vector__ void row_softmax_persistent_forward_kernel" in source
    assert "__global__ __vector__ void row_softmax_fast_forward_kernel" in source
    assert "__global__ __vector__ void row_softmax_fast_reg_forward_kernel" in source
    assert "__global__ __vector__ void cunn_spatial_softmax_forward_kernel" in source
    assert "asc_vf_call<row_softmax_persistent_forward_vf" in source
    assert "asc_vf_call<row_softmax_fast_forward_vf" in source
    assert "asc_vf_call<row_softmax_fast_reg_forward_vf" in source
    assert "asc_vf_call<cunn_spatial_softmax_forward_vf" in source
    assert "<<<grid_x, 0, acl_stream>>>" in source or "<<<grid_x,\n         0,\n         acl_stream>>>" in source
    assert "<<<grid_x, dynamic_ubuf_bytes, acl_stream>>>" in source or "<<<grid_x,\n         dynamic_ubuf_bytes,\n         acl_stream>>>" in source
    assert "<<<launch.grid_x * launch.grid_y," in source
    assert "launch.dynamic_ubuf_bytes" in source


def test_ascend_softmax_v3_rowwise_grid_cap_is_shape_aware():
    source = _read_v3_simt_source("spatial_softmax.asc")
    rowwise_start = source.index("inline int64_t row_softmax_grid_x")
    rowwise_end = source.index("inline const char* row_softmax_api_name")
    rowwise_source = source[rowwise_start:rowwise_end]

    launch_start = source.index("void launch_row_forward_impl")
    launch_end = source.index("at::Tensor spatial_forward_common(")
    launch_source = source[launch_start:launch_end]

    assert "constexpr int64_t kRowSoftmaxPhysicalGridXLimit = 64" in rowwise_source
    assert "constexpr int64_t kRowSoftmaxGridXLimit = 32768" in rowwise_source
    assert "row_softmax_persistent_grid_x(outer_size, block_y, warp_batch, dim_size)" in launch_source
    assert "row_softmax_grid_x(outer_size)" in launch_source
    assert "dim_size <= 256 || dim_size == 512 || dim_size == 1024" in rowwise_source


def test_ascend_softmax_v2_fast_path_has_fp16_half2_x4_vector_path():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    fast_start = source.index("row_softmax_fast_forward_kernel")
    generic_start = source.index("row_softmax_generic_forward_kernel")
    fast_source = source[fast_start:generic_start]

    assert "fast_fp16x4_reduce" in source
    assert "fast_fp16x4_write" in source
    assert "half2" in source
    assert "__low2float" in source
    assert "__high2float" in source
    assert "__floats2half2_rn" in source
    assert "dim_size % 4 == 0" in fast_source
    assert "std::is_same_v<scalar_t, __fp16>" in fast_source


def test_ascend_softmax_v2_generic_path_is_explicitly_unimplemented():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    generic_start = source.index("row_softmax_generic_forward_kernel")
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    generic_source = source[generic_start:spatial_start]

    assert "row_softmax_generic_forward_kernel" in source
    assert "path == RowSoftmaxPath::GenericLike" in source
    assert "aten_softmax_v2::row_softmax_generic_forward" in source
    assert "generic row-wise softmax is not implemented or accuracy-verified" in source
    assert "spatial_block_reduce_x" not in generic_source
    assert "output[row_offset + d]" not in generic_source


def test_ascend_softmax_v2_spatial_path_keeps_cuda_style_launch_policy():
    header = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "occupancy_common.h"
    ).read_text()
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()

    assert "inner_threads = std::min(inner_size, kMaxThreads)" in header
    assert "inner_threads <= 64 && dim_size >= 64" in header
    assert "inner_blocks = ceil_div(inner_size, block_y)" in header
    assert "outer_blocks = ceil_div(max_active_blocks, inner_blocks)" in header
    assert "block_x == 1 ? 0 : block_threads * accscalar_size" in header
    assert "cunn_spatial_softmax_forward_kernel" in source
    assert "spatial_block_reduce_x" in source
    assert "blockIdx.y * blockDim.y + threadIdx.y" in source


def test_ascend_softmax_v2_spatial_kernel_matches_cuda_layout_shape():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    launch_start = source.index("launch_spatial_forward_impl")
    spatial_source = source[spatial_start:launch_start]

    assert "const int64_t outer_stride = inner_size * dim_size" in spatial_source
    assert "const int64_t dim_stride = inner_size" in spatial_source
    assert "outer_offset = outer_index * outer_stride" in spatial_source
    assert "inner_index = blockIdx.y * blockDim.y + threadIdx.y" in spatial_source
    assert "data_offset = outer_offset + inner_index" in spatial_source
    assert "if (blockDim.x > 1)" in spatial_source
    assert "} else {" in spatial_source
    assert "shared_offset" not in spatial_source
    assert "input[data_offset + d * dim_stride]" in spatial_source
    assert "output[data_offset + d * dim_stride]" in spatial_source
    assert "spatial_block_reduce_x<accscalar_t, Max>" in spatial_source
    assert "spatial_block_reduce_x<accscalar_t, Add>" in spatial_source


def test_ascend_softmax_v2_spatial_serial_branch_keeps_cuda_loop_shape():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    launch_start = source.index("launch_spatial_forward_impl")
    spatial_source = source[spatial_start:launch_start]
    serial_source = spatial_source[spatial_source.index("} else {") :]

    assert "for (int64_t d = threadIdx.x; d < dim_size; d += blockDim.x)" in serial_source
    assert "for (int64_t d = 0; d < dim_size; ++d)" not in serial_source


def test_ascend_softmax_v2_spatial_reduce_helper_applies_y_offset_like_cuda():
    source = (
        SIMT_OP_V2_ROOT
        / "aten_softmax_v2"
        / "csrc"
        / "simt"
        / "spatial_softmax.asc"
    ).read_text()
    helper_start = source.index("spatial_block_reduce_x")
    row_reduce_start = source.index("row_block_reduce")
    helper_source = source[helper_start:row_reduce_start]
    spatial_start = source.index("cunn_spatial_softmax_forward_kernel")
    launch_start = source.index("launch_spatial_forward_impl")
    spatial_source = source[spatial_start:launch_start]

    assert "shared += threadIdx.y * blockDim.x" in helper_source
    assert "shared_offset" not in helper_source
    assert "shared_offset" not in spatial_source


def test_ascend_softmax_accuracy_script_targets_v2_by_default():
    source = (
        Path("src/cannbench/operators/builtin/softmax/simt/test")
        / "ascend_softmax_accuracy.py"
    ).read_text()

    assert 'parser.add_argument("--simt-package", default="aten_softmax_v2")' in source
    assert 'parser.add_argument("--simt-label", default="simt_v2")' in source
    assert "from aten_softmax import ops" not in source

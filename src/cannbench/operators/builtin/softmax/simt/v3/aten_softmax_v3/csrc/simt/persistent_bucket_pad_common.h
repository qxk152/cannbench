#pragma once

#include <cstdint>
#include <limits>

#include "c_api/asc_simd.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"

namespace {

constexpr uint32_t kPersistentBucketWarpSize = 32;
constexpr uint32_t kPersistentBucketThreadsPerBlock = 1024;
constexpr uint32_t kPersistentBucketRowsPerTile =
    kPersistentBucketThreadsPerBlock / kPersistentBucketWarpSize;

template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    uint32_t kBucketCols>
__simt_vf__ __launch_bounds__(kPersistentBucketThreadsPerBlock) inline void
row_softmax_persistent_bucket_pad_vf(
    __ubuf__ outscalar_t* output_ub,
    __ubuf__ const scalar_t* input_ub,
    uint32_t row_count,
    uint32_t dim_size) {
  constexpr uint32_t kWarpIterations =
      (kBucketCols + kPersistentBucketWarpSize - 1) /
      kPersistentBucketWarpSize;
  const uint32_t row_in_tile = threadIdx.x / kPersistentBucketWarpSize;
  const uint32_t lane = threadIdx.x % kPersistentBucketWarpSize;
  if (row_in_tile >= row_count) {
    return;
  }

  const uint32_t row_offset = row_in_tile * kBucketCols;
  accscalar_t elements[kWarpIterations];

#pragma unroll
  for (uint32_t iter = 0; iter < kWarpIterations; ++iter) {
    const uint32_t col = lane + iter * kPersistentBucketWarpSize;
    elements[iter] = col < dim_size
        ? static_cast<accscalar_t>(input_ub[row_offset + col])
        : std::numeric_limits<accscalar_t>::lowest();
  }

  accscalar_t max_value = elements[0];
#pragma unroll
  for (uint32_t iter = 1; iter < kWarpIterations; ++iter) {
    max_value = max(max_value, elements[iter]);
  }
  max_value = asc_reduce_max(max_value);

  accscalar_t sum_value = 0;
#pragma unroll
  for (uint32_t iter = 0; iter < kWarpIterations; ++iter) {
    elements[iter] = __expf(elements[iter] - max_value);
    sum_value += elements[iter];
  }
  sum_value = asc_reduce_add(sum_value);

#pragma unroll
  for (uint32_t iter = 0; iter < kWarpIterations; ++iter) {
    const uint32_t col = lane + iter * kPersistentBucketWarpSize;
    output_ub[row_offset + col] =
        static_cast<outscalar_t>(elements[iter] / sum_value);
  }
}

template <typename scalar_t>
__aicore__ inline __gm__ scalar_t* mutable_gm_ptr(
    __gm__ const scalar_t* ptr) {
  return const_cast<__gm__ scalar_t*>(ptr);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    uint32_t kBucketCols>
__aicore__ inline void row_softmax_persistent_bucket_pad_pipeline_impl(
    __gm__ outscalar_t* output,
    __gm__ const scalar_t* input,
    int64_t outer_size,
    int64_t dim_size) {
  asc_init();

  __ubuf__ scalar_t input_tile_ub
      [2][kPersistentBucketRowsPerTile * kBucketCols];
  __ubuf__ outscalar_t output_tile_ub
      [2][kPersistentBucketRowsPerTile * kBucketCols];

  const uint32_t rows = static_cast<uint32_t>(outer_size);
  const uint32_t cols = static_cast<uint32_t>(dim_size);
  const uint32_t tile_stride =
      static_cast<uint32_t>(gridDim.x) * kPersistentBucketRowsPerTile;
  const uint32_t first_row_base =
      static_cast<uint32_t>(blockIdx.x) * kPersistentBucketRowsPerTile;
  const uint32_t tile_count = first_row_base < rows
      ? (rows - first_row_base + tile_stride - 1) / tile_stride
      : 0;
  const uint32_t row_bytes = cols * sizeof(scalar_t);
  const uint32_t output_row_bytes = cols * sizeof(outscalar_t);
  const uint32_t padded_input_row_bytes = kBucketCols * sizeof(scalar_t);
  const uint32_t padded_output_row_bytes = kBucketCols * sizeof(outscalar_t);
  const uint8_t input_right_padding_elems = static_cast<uint8_t>(
      ((32 - (row_bytes & 31)) & 31) / sizeof(scalar_t));

  asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
  asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID1);
  asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
  asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID1);

  for (uint32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int32_t event_id = (tile_idx & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
    const uint32_t slot = tile_idx & 1;
    const uint32_t row_base = first_row_base + tile_idx * tile_stride;
    const uint32_t rows_remaining = rows - row_base;
    const uint32_t row_count =
        rows_remaining >= kPersistentBucketRowsPerTile
        ? kPersistentBucketRowsPerTile
        : rows_remaining;

    asc_sync_wait(PIPE_V, PIPE_MTE2, event_id);
    asc_copy_gm2ub_align(
        input_tile_ub[slot],
        mutable_gm_ptr(input + static_cast<int64_t>(row_base) * cols),
        static_cast<uint16_t>(row_count),
        row_bytes,
        0,
        input_right_padding_elems,
        true,
        asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
        row_bytes,
        padded_input_row_bytes);
    asc_sync_notify(PIPE_MTE2, PIPE_V, event_id);

    asc_sync_wait(PIPE_MTE2, PIPE_V, event_id);
    asc_sync_wait(PIPE_MTE3, PIPE_V, event_id);
    asc_vf_call<row_softmax_persistent_bucket_pad_vf<
        scalar_t,
        accscalar_t,
        outscalar_t,
        kBucketCols>>(
        dim3(kPersistentBucketThreadsPerBlock),
        output_tile_ub[slot],
        input_tile_ub[slot],
        row_count,
        cols);
    asc_sync_notify(PIPE_V, PIPE_MTE2, event_id);

    asc_sync_notify(PIPE_V, PIPE_MTE3, event_id);
    asc_sync_wait(PIPE_V, PIPE_MTE3, event_id);
    asc_copy_ub2gm_align(
        output + static_cast<int64_t>(row_base) * cols,
        output_tile_ub[slot],
        static_cast<uint16_t>(row_count),
        output_row_bytes,
        asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
        output_row_bytes,
        padded_output_row_bytes);
    asc_sync_notify(PIPE_MTE3, PIPE_V, event_id);
  }

  asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);
  asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID1);
  asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
  asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    uint32_t kBucketCols>
__global__ __vector__ void row_softmax_persistent_bucket_pad_kernel(
    __gm__ outscalar_t* output,
    __gm__ const scalar_t* input,
    int64_t outer_size,
    int64_t dim_size) {
  row_softmax_persistent_bucket_pad_pipeline_impl<
      scalar_t,
      accscalar_t,
      outscalar_t,
      kBucketCols>(output, input, outer_size, dim_size);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    uint32_t kBucketCols>
void launch_row_softmax_persistent_bucket_pad_kernel(
    const scalar_t* input_ptr,
    outscalar_t* output_ptr,
    int64_t outer_size,
    int64_t dim_size,
    int64_t grid_x,
    void* acl_stream) {
  row_softmax_persistent_bucket_pad_kernel<
      scalar_t,
      accscalar_t,
      outscalar_t,
      kBucketCols>
      <<<grid_x, 0, acl_stream>>>(
          output_ptr,
          input_ptr,
          outer_size,
          dim_size);
}

} // namespace

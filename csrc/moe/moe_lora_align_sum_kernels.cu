#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"
#include "core/math.hpp"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace {

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                         int32_t col) {
  return row * total_col + col;
}

}  // namespace

// TODO: Refactor common parts with moe_align_sum_kernels
template <typename scalar_t, typename token_cnts_t>
__global__ void moe_lora_align_sum_kernel(
    scalar_t* __restrict__ topk_ids, scalar_t* __restrict__ token_lora_mapping,
    int64_t block_size, int num_experts, int max_loras, size_t numel,
    int max_num_tokens_padded, int max_num_m_blocks,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int topk_num, int32_t* total_tokens_post_pad, int32_t* num_tokens_per_lora, int32_t* adapter_enabled) {
  const size_t tokens_per_thread = div_ceil(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  int lora_id = blockIdx.x;
  if (adapter_enabled[lora_id] * num_tokens_per_lora[lora_id] == 0) {
    return;
  }
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  token_cnts_t* tokens_cnts = (token_cnts_t*)(shared_mem + num_experts + 1);

  // Initialize sorted_token_ids with numel
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[lora_id * max_num_tokens_padded + it] = numel;
  }

  // Initialize expert_ids with -1
  for (size_t it = threadIdx.x; it < max_num_m_blocks; it += blockDim.x) {
    expert_ids[lora_id * max_num_m_blocks + it] = -1;
  }

  // Initialize total_tokens_post_pad with 0
  if (threadIdx.x == 0) {
    total_tokens_post_pad[lora_id] = 0;
  }

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int mask = token_lora_mapping[i / topk_num] == lora_id;
    int idx = index(num_experts, threadIdx.x + 1, topk_ids[i]);
    tokens_cnts[idx] += mask;
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] +=
          tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  div_ceil(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                           block_size) *
                      block_size;
    }
    total_tokens_post_pad[lora_id] = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[index(max_num_m_blocks, lora_id, i / block_size)] =
          threadIdx.x;
    }
  }

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];

    int mask = (int)token_lora_mapping[i / topk_num] == lora_id;
    atomicAdd(
        &sorted_token_ids[index(max_num_tokens_padded, lora_id, rank_post_pad)],
        (i - numel) * mask);
    tokens_cnts[index(num_experts, threadIdx.x, expert_id)] += mask;
  }
}

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_lora_align_sum_kernel_large_num_experts(
    scalar_t* __restrict__ topk_ids, scalar_t* __restrict__ token_lora_mapping,
    int64_t block_size, int num_experts, int max_loras, size_t numel,
    int max_num_tokens_padded, int max_num_m_blocks,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int topk_num, int32_t* total_tokens_post_pad, int32_t* num_tokens_per_lora, int32_t* adapter_enabled,
    int32_t* __restrict__ cumsum, int32_t experts_per_warp, int32_t padded_num_experts) {
  

  int lora_id = blockIdx.x;
  if (adapter_enabled[lora_id] * num_tokens_per_lora[lora_id] == 0) {
    return;
  }

  // Initialize sorted_token_ids with numel
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[index(max_num_tokens_padded, lora_id, it)] = numel;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  
  extern __shared__ int32_t token_counts[];
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      token_counts[warp_id * experts_per_warp + i] = 0;
    }
  }
  // TODO(gnovack) - add lora masking
  // for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
  //   int mask = token_lora_mapping[i / topk_num] == lora_id;
  //   int idx = index(num_experts, threadIdx.x + 1, topk_ids[i]);
  //   tokens_cnts[idx] += mask;
  // }


  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts || token_lora_mapping[i / topk_num] != lora_id) {
      continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&token_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = token_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = div_ceil(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[(num_experts+1) * lora_id + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    *total_tokens_post_pad = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[(num_experts+1) * lora_id + threadIdx.x]; i < cumsum[(num_experts+1) * lora_id + threadIdx.x + 1];
         i += block_size) {
      expert_ids[index(max_num_m_blocks, lora_id, i / block_size)] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with 0
  // const size_t fill_start_idx = cumsum[index(num_experts+1, lora_id, num_experts)] / block_size + threadIdx.x;
  // const size_t expert_ids_size = div_ceil(max_num_tokens_padded, block_size);
  // for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x) {
  //   expert_ids[index(max_num_m_blocks, lora_id, i)] = 0;
  // }
}

template <typename scalar_t>
__global__ void lora_count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    size_t numel, int32_t num_experts, int max_num_tokens_padded) {
  
  const size_t lora_id = blockIdx.x;
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[index(num_experts+1, lora_id, expert_id)], 1);
    sorted_token_ids[index(max_num_tokens_padded, lora_id, rank_post_pad)] = i;
  }
}

void moe_lora_align_block_size(torch::Tensor topk_ids,
                               torch::Tensor token_lora_mapping,
                               int64_t num_experts, int64_t block_size,
                               int64_t max_loras, int64_t max_num_tokens_padded,
                               int64_t max_num_m_blocks,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor num_tokens_post_pad,
                               torch::Tensor num_tokens_per_lora,
                               torch::Tensor adapter_enabled) {
  const int topk_num = topk_ids.size(1);

  TORCH_CHECK(block_size > 0, "block_size should be greater than 0. ");

  auto dev = topk_ids.get_device();  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();


  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {

        bool small_batch_expert_mode = (topk_ids.numel() < 1024) && (num_experts <= 64);
        // bool small_batch_expert_mode = true;
        

        if (small_batch_expert_mode) {
          auto kernel = moe_lora_align_sum_kernel<scalar_t, int32_t>;
          
          const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);
          TORCH_CHECK(num_thread <= 1024,
              "num_thread must be less than 1024, "
              "and fallback is not implemented yet.");
          dim3 blockDim(num_thread);
          
          // Get max shared memory
          int device_max_shared_mem;
          cudaDeviceGetAttribute(&device_max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
          
          const int32_t shared_mem_size = ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
          if (shared_mem_size > device_max_shared_mem) {
            TORCH_CHECK(false,
                        "Shared memory usage exceeds device limit, and global memory "
                        "fallback is not implemented yet.");
          }

          AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize((void*)kernel, shared_mem_size));
          
          kernel<<<max_loras, blockDim, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              token_lora_mapping.data_ptr<scalar_t>(), block_size, num_experts,
              max_loras, topk_ids.numel(), max_num_tokens_padded,
              max_num_m_blocks, sorted_token_ids.data_ptr<int32_t>(),
              expert_ids.data_ptr<int32_t>(), topk_num,
              num_tokens_post_pad.data_ptr<int32_t>(), 
              num_tokens_per_lora.data_ptr<int32_t>(), 
              adapter_enabled.data_ptr<int32_t>()
            );
        } else {
          auto kernel = moe_lora_align_sum_kernel_large_num_experts<scalar_t, int32_t>;
          
          int num_thread = 1024;
          dim3 blockDim(num_thread);
          
          int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
          size_t num_warps = CEILDIV(padded_num_experts, WARP_SIZE);

          size_t shared_mem_size = num_warps * WARP_SIZE * sizeof(int32_t);

          // cumsum buffer
          auto options_int = torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
          torch::Tensor cumsum = torch::zeros({max_loras * (num_experts + 1)}, options_int);
          // torch::Tensor cumsum = torch::empty({max_loras * (num_experts + 1)}, options_int);

          kernel<<<max_loras, blockDim, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              token_lora_mapping.data_ptr<scalar_t>(), block_size, num_experts,
              max_loras, topk_ids.numel(), max_num_tokens_padded,
              max_num_m_blocks, sorted_token_ids.data_ptr<int32_t>(),
              expert_ids.data_ptr<int32_t>(), topk_num,
              num_tokens_post_pad.data_ptr<int32_t>(), 
              num_tokens_per_lora.data_ptr<int32_t>(), 
              adapter_enabled.data_ptr<int32_t>(),
              cumsum.data_ptr<int32_t>(),
              WARP_SIZE,
              padded_num_experts
            );

          const int block_threads = std::min(256, (int)num_thread);
          const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;

          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          dim3 gridDims(max_loras, actual_blocks);
          auto sort_kernel = lora_count_and_sort_expert_tokens_kernel<scalar_t>;

          // printf("Sorted tokens size: %d\n", sorted_token_ids.size(0));
          // printf("Max num tokens padded: %d\n", sorted_token_ids.size(0) / max_loras);

          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum.data_ptr<int32_t>(), 
            topk_ids.numel(), 
            num_experts,
            max_num_tokens_padded
          );
          
        }        
      });
}
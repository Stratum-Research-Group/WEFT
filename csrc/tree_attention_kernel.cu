#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel: each thread computes one row i of the tree mask
// mask[i, j] = 0.0f if j == i or j is an ancestor of i; else -inf
__global__ void build_causal_tree_mask_kernel(
    const int64_t* __restrict__ parent_indices,
    float* __restrict__ mask,
    int num_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    // Pointer to row i in the flattened [num_nodes, num_nodes] matrix
    float* row = mask + (i * num_nodes);

    // Initialize entire row to -INFINITY
    for (int j = 0; j < num_nodes; ++j) {
        row[j] = -1e9f;
    }

    // Node always attends to itself
    row[i] = 0.0f;

    // Traverse upwards via parent pointers
    int curr = static_cast<int>(parent_indices[i]);
    while (curr != -1 && curr >= 0 && curr < num_nodes) {
        row[curr] = 0.0f;
        curr = static_cast<int>(parent_indices[curr]);
    }
}

torch::Tensor build_causal_tree_mask_cuda(
    torch::Tensor parent_indices,
    int num_nodes
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(parent_indices.device());
    auto mask = torch::empty({num_nodes, num_nodes}, options);

    const int threads = 64;
    const int blocks = (num_nodes + threads - 1) / threads;

    build_causal_tree_mask_kernel<<<blocks, threads>>>(
        parent_indices.data_ptr<int64_t>(),
        mask.data_ptr<float>(),
        num_nodes
    );

    return mask;
}

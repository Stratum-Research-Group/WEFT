#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA launcher
torch::Tensor build_causal_tree_mask_cuda(
    torch::Tensor parent_indices,
    int num_nodes
);

torch::Tensor build_causal_tree_mask(
    torch::Tensor parent_indices,
    int num_nodes
) {
    TORCH_CHECK(parent_indices.is_cuda(), "parent_indices must be a CUDA tensor");
    TORCH_CHECK(parent_indices.dim() == 1, "parent_indices must be 1D");
    return build_causal_tree_mask_cuda(parent_indices, num_nodes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_causal_tree_mask", &build_causal_tree_mask, "Construct Causal Tree Attention Mask on CUDA");
}

"""
WEFT Fused Forward-KL Divergence & Soft-Target Cross-Entropy Loss Triton 3.9 Kernel.
Milestone Paper #140.
Computes temperature-scaled distillation loss:
L_KD = T^2 * sum_v P_T(v) * (log P_T(v) - log P_S(v))
directly in GPU SRAM across vocabulary blocks, eliminating intermediate probability allocation.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_kl_divergence_kernel(
    p_teacher_ptr,   # [B, V] float32 probabilities
    logits_stud_ptr, # [B, V] float32 student unnormalized logits
    loss_out_ptr,    # [B] float32 scalar loss per token
    temperature: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr
):
    pid_b = tl.program_id(0)

    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    offset = pid_b * V + offs_v

    # Load teacher prob and student logit
    p_t = tl.load(p_teacher_ptr + offset, mask=mask_v, other=1e-8)
    logits_s = tl.load(logits_stud_ptr + offset, mask=mask_v, other=-1e4)

    # Scale student logits by temperature
    scaled_logits = logits_s / temperature

    # Max-subtraction for numerical stability
    max_logit = tl.max(scaled_logits, axis=0)
    exp_logits = tl.exp(scaled_logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = max_logit + tl.log(sum_exp)

    # Log-softmax of student
    log_p_s = scaled_logits - log_sum_exp

    # Teacher log prob
    log_p_t = tl.log(p_t + 1e-8)

    # KL Divergence: P_T * (log P_T - log P_S)
    kl_terms = p_t * (log_p_t - log_p_s)
    kl_sum = tl.sum(kl_terms, axis=0)

    # Multiply by T^2 scaling factor
    loss = kl_sum * (temperature * temperature)

    tl.store(loss_out_ptr + pid_b, loss)

def fused_distillation_kl_triton(
    p_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
    """
    Computes fused KL divergence loss on CUDA GPU.
    p_teacher: [B, V]
    logits_student: [B, V]
    """
    assert p_teacher.is_cuda and logits_student.is_cuda, "Inputs must be on CUDA GPU"
    assert p_teacher.shape == logits_student.shape, "Shapes must match"

    B, V = p_teacher.shape
    loss_out = torch.empty((B,), dtype=torch.float32, device=p_teacher.device)

    BLOCK_V = triton.next_power_of_2(V)
    grid = (B,)

    _fused_kl_divergence_kernel[grid](
        p_teacher.contiguous(),
        logits_student.contiguous(),
        loss_out,
        temperature=float(temperature),
        V=V,
        BLOCK_V=BLOCK_V
    )
    return loss_out.mean()

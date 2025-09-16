import torch
import triton


def fused_gemm_topk(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    k: int,
):
    """
    Mathematically, equates to:
        y = x @ W.T + b
    Before setting all but the topK elements of y to 0, and returning y.

    Params:
        x: (M, d_hidden)
        W: (d_sae, d_hidden)
        b: (d_sae)
        k: int

    Returns:
        y: (M, d_sae)
    """


@triton.jit
def _fused_gemm_topk_kernel():
    pass

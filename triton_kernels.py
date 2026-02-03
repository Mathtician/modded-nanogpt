import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Block-diagonal c_proj layout (3 x 1024 x 256)
NUM_CPROJ_BLOCKS = 3
CPROJ_BLOCK_IN = 1024
CPROJ_BLOCK_OUT = 256
CPROJ_ROWS = NUM_CPROJ_BLOCKS * CPROJ_BLOCK_IN   # 3072
CPROJ_COLS = NUM_CPROJ_BLOCKS * CPROJ_BLOCK_OUT  # 768
CPROJ_ASSERT = bool(int(os.environ.get("CPROJ_ASSERT", "0")))

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded configs based on H100 autotuning
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
        num_stages, num_warps = 4, 4
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
        num_stages, num_warps = 4, 4

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded config based on H100 autotuning (M=768)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    num_stages, num_warps = 4, 4

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

# -----------------------------------------------------------------------------
# Block-diagonal matmuls for MLP c_proj (3 blocks of 1024x256)

@triton.jit
def cproj_block_diag_fwd_kernel(
    A_ptr, B_ptr, C_ptr,
    M,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CPROJ_BLOCK_IN: tl.constexpr,
    CPROJ_BLOCK_OUT: tl.constexpr,
    NUM_CPROJ_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(CPROJ_BLOCK_OUT, BLOCK_SIZE_N)

    block_id = pid_bn // num_pid_n
    pid_n = pid_bn % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(CPROJ_BLOCK_IN, BLOCK_SIZE_K)):
        k_offsets = block_id * CPROJ_BLOCK_IN + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + (block_id * CPROJ_BLOCK_OUT + offs_n)[None, :] * stride_bn

        k_mask = k_offsets[None, :] < (block_id + 1) * CPROJ_BLOCK_IN
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask.T & (offs_n[None, :] < CPROJ_BLOCK_OUT), other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    output = accumulator.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + (block_id * CPROJ_BLOCK_OUT + offs_n)[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < CPROJ_BLOCK_OUT)
    tl.store(c_ptrs, output, mask=c_mask)


def cproj_block_diag_fwd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B where B is block-diagonal with 3 blocks of size 1024x256.
    A: (..., M, 3072)
    B: (3072, 768)
    Returns: (..., M, 768)
    """
    assert A.shape[-1] == CPROJ_ROWS
    assert B.shape == (CPROJ_ROWS, CPROJ_COLS)

    M = A.shape[-2]
    C = torch.empty((*A.shape[:-1], CPROJ_COLS), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    num_warps = 4
    num_stages = 4

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        NUM_CPROJ_BLOCKS * triton.cdiv(CPROJ_BLOCK_OUT, BLOCK_SIZE_N),
    )
    cproj_block_diag_fwd_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        stride_am=A.stride(-2),
        stride_ak=A.stride(-1),
        stride_bk=B.stride(0),
        stride_bn=B.stride(1),
        stride_cm=C.stride(-2),
        stride_cn=C.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        CPROJ_BLOCK_IN=CPROJ_BLOCK_IN,
        CPROJ_BLOCK_OUT=CPROJ_BLOCK_OUT,
        NUM_CPROJ_BLOCKS=NUM_CPROJ_BLOCKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


@triton.jit
def cproj_block_diag_bwd_input_kernel(
    G_ptr, W_ptr, PRE_ptr, D_ptr,
    M,
    stride_gm, stride_gn,
    stride_wk, stride_wn,
    stride_pm, stride_pk,
    stride_dm, stride_dk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CPROJ_BLOCK_IN: tl.constexpr,
    CPROJ_BLOCK_OUT: tl.constexpr,
    NUM_CPROJ_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(CPROJ_BLOCK_IN, BLOCK_SIZE_N)

    block_id = pid_bn // num_pid_n
    pid_n = pid_bn % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    g_ptrs = G_ptr + offs_m[:, None] * stride_gm + (block_id * CPROJ_BLOCK_OUT + offs_k[None, :]) * stride_gn
    w_ptrs = W_ptr + (block_id * CPROJ_BLOCK_IN + offs_n[None, :]) * stride_wk + (block_id * CPROJ_BLOCK_OUT + offs_k[:, None]) * stride_wn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(CPROJ_BLOCK_OUT, BLOCK_SIZE_K)):
        g = tl.load(g_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < CPROJ_BLOCK_OUT), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < CPROJ_BLOCK_OUT) & (offs_n[None, :] < CPROJ_BLOCK_IN), other=0.0)
        acc = tl.dot(g, w, acc)
        g_ptrs += BLOCK_SIZE_K * stride_gn
        w_ptrs += BLOCK_SIZE_K * stride_wk
        offs_k += BLOCK_SIZE_K

    # Apply ReLU^2 derivative with saved pre-activation
    pre_ptrs = PRE_ptr + offs_m[:, None] * stride_pm + (block_id * CPROJ_BLOCK_IN + offs_n[None, :]) * stride_pk
    pre = tl.load(pre_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < CPROJ_BLOCK_IN), other=0.0)
    gated = 2 * acc * tl.where(pre > 0, pre, 0).to(acc.dtype)

    out = gated.to(D_ptr.dtype.element_ty)
    d_ptrs = D_ptr + offs_m[:, None] * stride_dm + (block_id * CPROJ_BLOCK_IN + offs_n[None, :]) * stride_dk
    d_mask = (offs_m[:, None] < M) & (offs_n[None, :] < CPROJ_BLOCK_IN)
    tl.store(d_ptrs, out, mask=d_mask)


def cproj_block_diag_bwd_input(grad_out: torch.Tensor, weight: torch.Tensor, pre: torch.Tensor):
    """
    Compute dpre for block-diagonal c_proj: grad_out @ weight.T with ReLU^2 derivative.
    grad_out: (..., M, 768)
    weight:   (3072, 768)
    pre:      (..., M, 3072)
    returns:  (..., M, 3072)
    """
    M = grad_out.shape[-2]
    assert grad_out.shape[-1] == CPROJ_COLS
    assert weight.shape == (CPROJ_ROWS, CPROJ_COLS)
    assert pre.shape[-1] == CPROJ_ROWS

    dpre = torch.zeros_like(pre)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    num_warps = 4
    num_stages = 4

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        NUM_CPROJ_BLOCKS * triton.cdiv(CPROJ_BLOCK_IN, BLOCK_SIZE_N),
    )
    cproj_block_diag_bwd_input_kernel[grid](
        G_ptr=grad_out,
        W_ptr=weight,
        PRE_ptr=pre,
        D_ptr=dpre,
        M=M,
        stride_gm=grad_out.stride(-2),
        stride_gn=grad_out.stride(-1),
        stride_wk=weight.stride(0),
        stride_wn=weight.stride(1),
        stride_pm=pre.stride(-2),
        stride_pk=pre.stride(-1),
        stride_dm=dpre.stride(-2),
        stride_dk=dpre.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        CPROJ_BLOCK_IN=CPROJ_BLOCK_IN,
        CPROJ_BLOCK_OUT=CPROJ_BLOCK_OUT,
        NUM_CPROJ_BLOCKS=NUM_CPROJ_BLOCKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dpre


@triton.jit
def cproj_block_diag_dweight_kernel(
    POST_ptr, G_ptr, DW_ptr,
    M,
    stride_pm, stride_pk,
    stride_gm, stride_gn,
    stride_dk, stride_dn,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    CPROJ_BLOCK_IN: tl.constexpr,
    CPROJ_BLOCK_OUT: tl.constexpr,
    NUM_CPROJ_BLOCKS: tl.constexpr,
):
    # pid_k encodes block_id and row tile; pid_n encodes col tile
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_k = tl.cdiv(CPROJ_BLOCK_IN, BLOCK_SIZE_M)
    block_id = pid_k // num_pid_k
    pid_m = pid_k % num_pid_k

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    post_ptrs = POST_ptr + (offs_k[:, None] * stride_pm) + (block_id * CPROJ_BLOCK_IN + offs_m[None, :]) * stride_pk
    g_ptrs = G_ptr + (offs_k[:, None] * stride_gm) + (block_id * CPROJ_BLOCK_OUT + offs_n[None, :]) * stride_gn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        post = tl.load(post_ptrs, mask=(offs_k[:, None] < M) & (offs_m[None, :] < CPROJ_BLOCK_IN), other=0.0)
        gout = tl.load(g_ptrs, mask=(offs_k[:, None] < M) & (offs_n[None, :] < CPROJ_BLOCK_OUT), other=0.0)
        acc += tl.dot(post.T, gout)  # post is (K, M); transpose to (M, K) to match gout (K, N)
        post_ptrs += BLOCK_SIZE_K * stride_pm
        g_ptrs += BLOCK_SIZE_K * stride_gm
        offs_k += BLOCK_SIZE_K

    out = acc.to(DW_ptr.dtype.element_ty)
    dw_ptrs = DW_ptr + (block_id * CPROJ_BLOCK_IN + offs_m[:, None]) * stride_dk + (block_id * CPROJ_BLOCK_OUT + offs_n[None, :]) * stride_dn
    mask = (offs_m[:, None] < CPROJ_BLOCK_IN) & (offs_n[None, :] < CPROJ_BLOCK_OUT)
    tl.store(dw_ptrs, out, mask=mask)


def cproj_block_diag_dweight(post: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient for block-diagonal c_proj weights: post^T @ grad_out (diag blocks only).
    post: (..., M, 3072)
    grad_out: (..., M, 768)
    returns: (3072, 768) with only diag blocks filled
    """
    M = post.shape[-2]
    assert post.shape[-1] == CPROJ_ROWS
    assert grad_out.shape[-1] == CPROJ_COLS
    dw = torch.zeros((CPROJ_ROWS, CPROJ_COLS), device=post.device, dtype=post.dtype)

    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    num_warps = 4
    num_stages = 4

    grid = (
        NUM_CPROJ_BLOCKS * triton.cdiv(CPROJ_BLOCK_IN, BLOCK_SIZE_M),
        triton.cdiv(CPROJ_BLOCK_OUT, BLOCK_SIZE_N),
    )
    cproj_block_diag_dweight_kernel[grid](
        POST_ptr=post,
        G_ptr=grad_out,
        DW_ptr=dw,
        M=M,
        stride_pm=post.stride(-2),
        stride_pk=post.stride(-1),
        stride_gm=grad_out.stride(-2),
        stride_gn=grad_out.stride(-1),
        stride_dk=dw.stride(0),
        stride_dn=dw.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        CPROJ_BLOCK_IN=CPROJ_BLOCK_IN,
        CPROJ_BLOCK_OUT=CPROJ_BLOCK_OUT,
        NUM_CPROJ_BLOCKS=NUM_CPROJ_BLOCKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dw

def _assert_cproj_offblock_nan(tensor: torch.Tensor, name: str):
    if not CPROJ_ASSERT or torch._dynamo.is_compiling():
        return
    with torch.no_grad():
        for b in range(NUM_CPROJ_BLOCKS):
            r0 = b * CPROJ_BLOCK_IN
            r1 = r0 + CPROJ_BLOCK_IN
            c0 = b * CPROJ_BLOCK_OUT
            c1 = c0 + CPROJ_BLOCK_OUT
            left = tensor[..., r0:r1, :c0]
            right = tensor[..., r0:r1, c1:]
            if left.numel():
                torch._assert(torch.isnan(left).all(), f"{name}: off-block NaNs leaked (block {b}, left)")
            if right.numel():
                torch._assert(torch.isnan(right).all(), f"{name}: off-block NaNs leaked (block {b}, right)")

def _assert_cproj_offblock_zero(tensor: torch.Tensor, name: str):
    if not CPROJ_ASSERT or torch._dynamo.is_compiling():
        return
    with torch.no_grad():
        for b in range(NUM_CPROJ_BLOCKS):
            r0 = b * CPROJ_BLOCK_IN
            r1 = r0 + CPROJ_BLOCK_IN
            c0 = b * CPROJ_BLOCK_OUT
            c1 = c0 + CPROJ_BLOCK_OUT
            left = tensor[..., r0:r1, :c0]
            right = tensor[..., r0:r1, c1:]
            if left.numel():
                torch._assert((left == 0).all(), f"{name}: off-block grad not zero (block {b}, left)")
            if right.numel():
                torch._assert((right == 0).all(), f"{name}: off-block grad not zero (block {b}, right)")

def _assert_tensor_finite(tensor: torch.Tensor, name: str):
    if not CPROJ_ASSERT or torch._dynamo.is_compiling():
        return
    with torch.no_grad():
        torch._assert(torch.isfinite(tensor).all(), f"{name}: found non-finite values")

# -----------------------------------------------------------------------------
# Block-diagonal XXT and ba_plus_cAA (used by NorMuon on c_proj)

@triton.jit
def XXT_block_diag_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    NUM_CPROJ_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    block_rows = M // NUM_CPROJ_BLOCKS
    block_cols = K // NUM_CPROJ_BLOCKS

    block_m = m_idx // block_rows
    block_n = n_idx // block_rows
    if block_m != block_n:
        return  # Off-diagonal blocks are zero for block-diagonal input

    # Skip blocks that don't need to be computed (upper/lower tri)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_idx + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = block_m * block_cols + tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in tl.range(0, tl.cdiv(block_cols, BLOCK_SIZE_K)):
        a_ptrs = A_ptr + offs_m[:, None] * a_stride_r + k_offsets[None, :] * a_stride_c
        at_ptrs = A_ptr + k_offsets[:, None] * a_stride_c + offs_n[None, :] * a_stride_r
        k_mask = k_offsets[None, :] < (block_m + 1) * block_cols
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        at = tl.load(at_ptrs, mask=k_mask.T & (offs_n[None, :] < M), other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        k_offsets += BLOCK_SIZE_K

    output = accumulator.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * c_stride_r + offs_n[None, :] * c_stride_c
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)
    # Mirror across diagonal for symmetry
    c_ptrs_t = C_ptr + offs_n[:, None] * c_stride_r + offs_m[None, :] * c_stride_c
    tl.store(c_ptrs_t, output.T, mask=c_mask.T)


def XXT_block_diag(A: torch.Tensor, out: torch.Tensor):
    """
    Compute C = A @ A.T for block-diagonal A (3 blocks).
    Only diagonal blocks are computed.
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M and out.size(-1) == M
    assert M % NUM_CPROJ_BLOCKS == 0 and K % NUM_CPROJ_BLOCKS == 0

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    num_stages, num_warps = 4, 4

    out.zero_()
    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    XXT_block_diag_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        NUM_CPROJ_BLOCKS=NUM_CPROJ_BLOCKS,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out


@triton.jit
def ba_plus_cAA_block_diag_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    NUM_CPROJ_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    block_rows = M // NUM_CPROJ_BLOCKS
    block_cols = K // NUM_CPROJ_BLOCKS

    block_m = m_idx // block_rows
    block_n = n_idx // block_rows
    if block_m != block_n:
        return

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_idx + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = block_m * block_cols + tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in tl.range(0, tl.cdiv(block_cols, BLOCK_SIZE_K)):
        a_ptrs = A_ptr + offs_m[:, None] * a_stride_r + k_offsets[None, :] * a_stride_c
        at_ptrs = A_ptr + k_offsets[:, None] * a_stride_c + offs_n[None, :] * a_stride_r
        k_mask = k_offsets[None, :] < (block_m + 1) * block_cols
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        at = tl.load(at_ptrs, mask=k_mask.T & (offs_n[None, :] < M), other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        k_offsets += BLOCK_SIZE_K

    # Load block of A to add (corresponds to the current diagonal block of C)
    offs_am = offs_m
    offs_an = offs_n
    a_add_ptrs = A_ptr + offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += a_add * beta

    output = accumulator.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * c_stride_r + offs_n[None, :] * c_stride_c
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)
    c_ptrs_t = C_ptr + offs_n[:, None] * c_stride_r + offs_m[None, :] * c_stride_c
    tl.store(c_ptrs_t, output.T, mask=c_mask.T)


def ba_plus_cAA_block_diag(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Compute C = alpha * A @ A.T + beta * A for block-diagonal A (3 blocks).
    Only diagonal blocks are computed.
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M and out.size(-1) == M
    assert M % NUM_CPROJ_BLOCKS == 0 and K % NUM_CPROJ_BLOCKS == 0

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    num_stages, num_warps = 4, 4

    out.zero_()
    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    ba_plus_cAA_block_diag_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        NUM_CPROJ_BLOCKS=NUM_CPROJ_BLOCKS,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

# -----------------------------------------------------------------------------
# Triton kernel for MLP: relu(x @ W1.T)^2, by @andrewbriand, @jrauvola

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c], c0)

        if FORWARD:
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)

        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def linear_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    FORWARD = False
    if aux is None:
        FORWARD = True
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    linear_relu_square_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        num_stages=num_stages,
        num_warps=num_warps
    )

    if FORWARD:
        return c, aux
    else:
        return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        pre, post = linear_relu_square(x.view((-1, x.shape[-1])), W1)
        x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        dW2 = post.T @ grad_output
        dpre = linear_relu_square(grad_output.view((-1, grad_output.shape[-1])), W2, aux=pre)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2


class FusedLinearReLUSquareBlockDiagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        _assert_cproj_offblock_nan(W2, "c_proj_weight_forward")
        x_flat = x.view((-1, x.shape[-1]))
        pre, post = linear_relu_square(x_flat, W1)
        out = cproj_block_diag_fwd(post, W2)
        _assert_tensor_finite(out, "c_proj_forward_out")
        ctx.save_for_backward(x, W1, W2, pre, post)
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        grad_flat = grad_output.view((-1, grad_output.shape[-1]))
        _assert_tensor_finite(grad_flat, "c_proj_grad_out")
        dW2 = cproj_block_diag_dweight(post, grad_flat)
        dpre = cproj_block_diag_bwd_input(grad_flat, W2, pre)
        x_flat = x.view((-1, x.shape[-1]))
        dW1 = dpre.T @ x_flat
        dx = dpre @ W1
        _assert_cproj_offblock_zero(dW2, "c_proj_grad_weight")
        return dx.view(x.shape), dW1, dW2

# -----------------------------------------------------------------------------
# Fused Softcapped Cross Entropy


@triton.jit
def fused_softcapped_entropy_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n

    max_val = -float('inf')
    sum_exp = 0.0

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
        z = A * tl.sigmoid((val + B) / C)
        z = tl.where(mask, z, -float('inf'))
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + row_idx, lse)

    total_loss = 0.0
    for k in range(n_predict):
        target_idx = row_idx + k
        if target_idx < n_rows:
            weight = tl.load(mtp_weights_ptr + k)
            if weight > 0:
                target = tl.load(targets_ptr + target_idx).to(tl.int32)
                if target >= 0 and target < n_cols:
                    val_target = tl.load(logits_row_ptr + target).to(tl.float32)
                    z_target = A * tl.sigmoid((val_target + B) / C)
                    total_loss += weight * (lse - z_target)

    tl.store(losses_ptr + row_idx, total_loss)

@triton.jit
def fused_softcapped_entropy_bwd_kernel(
    grad_input_ptr, grad_output_ptr, lse_ptr, logits_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v, stride_grad_n, stride_grad_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)

    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_input_ptr + row_idx * stride_grad_n

    lse = tl.load(lse_ptr + row_idx)
    grad_loss = tl.load(grad_output_ptr + row_idx)

    S_w = 0.0
    for k in range(n_predict):
        if row_idx + k < n_rows:
            S_w += tl.load(mtp_weights_ptr + k)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        u = (val + B) / C
        sigmoid_u = tl.sigmoid(u)
        z = A * sigmoid_u
        p = tl.exp(z - lse)

        term1 = S_w * p
        term2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for k in range(n_predict):
            if row_idx + k < n_rows:
                target = tl.load(targets_ptr + row_idx + k).to(tl.int32)
                weight = tl.load(mtp_weights_ptr + k)
                term2 += tl.where(cols == target, weight, 0.0)

        grad_z = grad_loss * (term1 - term2)
        dz_dx = (1.0 / C) * z * (1.0 - sigmoid_u)
        grad_x = grad_z * dz_dx
        tl.store(grad_row_ptr + cols, grad_x.to(tl.bfloat16), mask=mask)

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, mtp_weights, A=23.0, B=5.0, C=7.5):
        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grid = (n_rows,)
        fused_softcapped_entropy_fwd_kernel[grid](
            logits, losses, lse, targets, mtp_weights,
            logits.stride(0), logits.stride(1),
            n_rows, n_cols, n_predict,
            A, B, C,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=4
        )

        ctx.save_for_backward(logits, targets, mtp_weights, lse)
        ctx.params = (A, B, C)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse = ctx.saved_tensors
        A, B, C = ctx.params
        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.bfloat16, device=logits.device)
        grad_output = grad_output.contiguous()

        grid = (n_rows,)
        fused_softcapped_entropy_bwd_kernel[grid](
            grad_input, grad_output, lse, logits, targets, mtp_weights,
            logits.stride(0), logits.stride(1), grad_input.stride(0), grad_input.stride(1),
            n_rows, n_cols, n_predict,
            A, B, C,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=4
        )
        return grad_input, None, None, None, None, None

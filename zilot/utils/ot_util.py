import math
from functools import partial

import torch


@torch.jit.script
def dualsort_init(M: torch.Tensor, iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """DualSort initializer for 1D Sinkhorn"""
    B, n, m = M.size()
    k = max(n, m)
    c = torch.zeros(B, k, k, device=M.device, dtype=M.dtype)
    c[:, :n, :m] = M
    c_m_c_diag_one_t = c - c.diag()[:, None, :]
    f = torch.zeros(B, k, device=M.device, dtype=M.dtype)
    for ii in range(iters):
        # f_i = min_j (c_ij - c_jj + f_j)
        f = torch.min(c_m_c_diag_one_t + f[:, None, :], dim=1).values
    if n < m:
        logu = torch.zeros(B, n, device=M.device, dtype=M.dtype)
        logv = f / eps
    else:
        logu = f / eps
        logv = torch.zeros(B, m, device=M.device, dtype=M.dtype)
    return logu, logv


@torch.jit.script
def sinkhorn_log_unbalanced(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    eps: float,
    iters: int,
    tau_a: float,
    tau_b: float,
):
    """Batched Sinkhorn algorithm with log-sum-exp stabilization for Unbalanced OT"""
    rho_a = 1.0 if tau_a == 1.0 else (tau_a / (1.0 - tau_a)) / ((tau_a / (1.0 - tau_a)) + eps)
    rho_b = 1.0 if tau_b == 1.0 else (tau_b / (1.0 - tau_b)) / ((tau_b / (1.0 - tau_b)) + eps)

    Mr = -M / eps
    a_zero = a == 0.0
    b_zero = b == 0.0
    loga = torch.where(a_zero, 1.0, a).log()
    logb = torch.where(b_zero, 1.0, b).log()
    logu = torch.zeros_like(loga)
    logv = torch.zeros_like(logb)

    for ii in range(iters):
        # update logu = rho_a * (loga - (-M/r) + logv)
        # update logv = rho_b * (logb - (-M/r)^T + logu)
        logu = rho_a * torch.where(a_zero, logu, (loga - torch.logsumexp(Mr + logv[:, None, :], dim=-1)))
        logv = rho_b * torch.where(b_zero, logv, (logb - torch.logsumexp(Mr + logu[:, :, None], dim=-2)))

    pi = torch.exp(logu[:, :, None] + Mr + logv[:, None, :])
    pi = torch.where(a_zero[:, :, None] | b_zero[:, None, :], 0.0, pi)
    cost = (pi * M).sum(dim=(-2, -1))

    return cost, pi


@torch.compile(mode="max-autotune")
def sinkhorn_fixed_size_superstep(
    loga: torch.Tensor,
    logb: torch.Tensor,
    a_zero: torch.Tensor,
    b_zero: torch.Tensor,
    Mr: torch.Tensor,
    n_steps: int,
    rho_a: float,
    rho_b: float,
    logu: torch.Tensor,
    logv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for _ in range(n_steps):
        logu = rho_a * torch.where(a_zero, logu, (loga - torch.logsumexp(Mr + logv[:, None, :], dim=-1)))
        logv = rho_b * torch.where(b_zero, logv, (logb - torch.logsumexp(Mr + logu[:, :, None], dim=-2)))
    return logu, logv


def sinkhorn_log_unbalanced_fixed(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    eps: float,
    iters: int,
    tau_a: float,
    tau_b: float,
):
    """Batched Sinkhorn algorithm with log-sum-exp stabilization for fixed-size OT"""

    # pad M, a, b to the next power of 2 to not trigger too many recompilations
    B, n, m = M.size()
    n_pow2 = max(16, 2 ** math.ceil(math.log2(n)))
    m_pow2 = max(16, 2 ** math.ceil(math.log2(m)))
    M = torch.nn.functional.pad(M, (0, m_pow2 - m, 0, n_pow2 - n), value=0.0)
    a = torch.nn.functional.pad(a, (0, n_pow2 - n), value=0.0)
    b = torch.nn.functional.pad(b, (0, m_pow2 - m), value=0.0)

    rho_a = 1.0 if tau_a == 1.0 else (tau_a / (1.0 - tau_a)) / ((tau_a / (1.0 - tau_a)) + eps)
    rho_b = 1.0 if tau_b == 1.0 else (tau_b / (1.0 - tau_b)) / ((tau_b / (1.0 - tau_b)) + eps)
    Mr = -M / eps
    a_zero = a == 0.0
    b_zero = b == 0.0
    loga = torch.where(a_zero, 1.0, a).log()
    logb = torch.where(b_zero, 1.0, b).log()
    superstep = 16
    iters = math.ceil(iters / superstep)
    logu = torch.zeros_like(loga)
    logv = torch.zeros_like(logb)
    step_fn = partial(sinkhorn_fixed_size_superstep, loga, logb, a_zero, b_zero, Mr, superstep, rho_a, rho_b)
    for _ in range(iters):
        logu, logv = step_fn(logu, logv)
        logu, logv = logu.clone(), logv.clone()
    pi = torch.exp(logu[:, :, None] + Mr + logv[:, None, :])
    pi = torch.where(a_zero[:, :, None] | b_zero[:, None, :], 0.0, pi)
    cost = (pi * M).sum(dim=(-2, -1))

    return cost, pi[:, :n, :m]


@torch.jit.script
def sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    logu: torch.Tensor,
    logv: torch.Tensor,
    eps: float,
    iters: int,
):
    """Batched Sinkhorn algorithm with log-sum-exp stabilization"""

    Mr = -M / eps
    loga = a.log()
    logb = b.log()

    for ii in range(iters):
        # update logv = logb - (-M/r)^T + logu
        # update logu = loga - (-M/r) + logv
        torch.logsumexp(Mr + logu[:, :, None], dim=-2, out=logv).neg_().add_(logb)
        torch.logsumexp(Mr + logv[:, None, :], dim=-1, out=logu).neg_().add_(loga)

    pi = torch.exp(logu[:, :, None] + Mr + logv[:, None, :])
    err = torch.norm(pi.sum(-2) - b, p=2, dim=-1)  # L2-norm marginal violation for b
    cost = (pi * M).sum(dim=(-2, -1))

    return cost, pi, err, logu, logv


@torch.jit.script
def northwest(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor):
    """Batched North-West Corner Rule for OT"""
    with torch.no_grad():  # this function is not differentiable
        B, n, m = M.size()
        k = n + m - 1
        a = a.clone()
        b = b.clone()
        i = torch.zeros(B, dtype=torch.long, device=M.device)
        j = torch.zeros(B, dtype=torch.long, device=M.device)
        gamma_idx = torch.empty(3, k, B, dtype=torch.int64, device=M.device).fill_(-1)
        gamma_idx[0, :] = torch.arange(B, device=gamma_idx.device, dtype=gamma_idx.dtype).expand(k, B)
        gamma_val = torch.empty(k, B, device=M.device, dtype=M.dtype)
        cost = torch.zeros(B, device=M.device, dtype=M.dtype)
        batch = torch.arange(B, device=M.device, dtype=torch.long)
        for it in range(k):
            amount = torch.minimum(a[batch, i], b[batch, j])
            gamma_idx[1, it] = i
            gamma_idx[2, it] = j
            gamma_val[it] = amount
            cost += amount * M[batch, i, j]
            move_i = torch.logical_or(a[batch, i] <= b[batch, j], j == m - 1)
            a[batch, i] = torch.where(move_i, 0.0, a[batch, i] - amount)
            b[batch, j] = torch.where(move_i, b[batch, j] - amount, 0.0)
            i = torch.where(move_i, i + 1, i)
            j = torch.where(move_i, j, j + 1)
        # assert gamma_idx[0].min() >= 0, f"gamma_idx[0] must be non-negative, {gamma_idx[0].min()}"
        # assert gamma_idx[0].max() < B, f"gamma_idx[0] must be less than B, {gamma_idx[0].max()}"
        # assert gamma_idx[1].min() >= 0, f"gamma_idx[1] must be non-negative, {gamma_idx[1].min()}"
        # assert gamma_idx[1].max() < n, f"gamma_idx[1] must be less than n, {gamma_idx[1].max()}"
        # assert gamma_idx[2].min() >= 0, f"gamma_idx[2] must be non-negative, {gamma_idx[2].min()}"
        # assert gamma_idx[2].max() < m, f"gamma_idx[2] must be less than m, {gamma_idx[2].max()}"
        gamma = torch.sparse_coo_tensor(gamma_idx.flatten(1), gamma_val.flatten(), M.size()).coalesce()
    return cost, gamma


# Dynamic Time Warping


@torch.jit.script
def _dtw(M: torch.Tensor, dp: torch.Tensor, ns: int, ms: int) -> torch.Tensor:
    """Batched Dynamic Time Warping"""
    with torch.no_grad():  # this function is not differentiable
        n, m = M.size(-2), M.size(-1)
        nn, mm = dp.size(-2), dp.size(-1)
        ne = ns + n
        me = ms + m
        assert ns > 0 and ms > 0, f"ns={ns} and ms={ms} must be positive"
        assert ne <= nn and me <= mm, f"ne={ne} > nn={nn} or me={me} > mm={mm} is out of bounds"
        for i in range(ns, ne):
            for j in range(ms, me):
                dp[..., i, j] = M[..., i - ns, j - ms] + torch.minimum(
                    dp[..., i - 1, j - 1], torch.minimum(dp[..., i - 1, j], dp[..., i, j - 1])
                )
    return dp[..., ne - 1, me - 1]


def dtw(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if M.ndim == 2:
        M = M.unsqueeze(0)
    B, n, m = M.size()
    dp = torch.empty((B, n + 1, m + 1), dtype=M.dtype, device=M.device)
    dp.fill_(torch.inf)
    dp[:, 0, 0] = 0.0
    cost = _dtw(M, dp, 1, 1)
    return cost, dp


def assert_dtw_dp_cond(dp: torch.Tensor):
    dp_ij = dp[..., :-1, :-1]
    dp_i = dp[..., :-1, 1:]
    dp_j = dp[..., 1:, :-1]
    dp_ij1 = dp[..., 1:, 1:]
    assert (dp_ij1 >= torch.minimum(dp_ij, torch.minimum(dp_i, dp_j))).all(), "DP condition not met"


@torch.jit.script
def dtw_backtrack(dp: torch.Tensor, start_i: int, start_j: int) -> torch.Tensor:
    """Backtrack the path for DTW"""
    with torch.no_grad():  # this function is not differentiable
        B, n, m = dp.size()
        path = torch.empty((2, n + m, B), dtype=torch.int64, device=dp.device).fill_(-1)
        i = torch.full((B,), start_i, dtype=torch.long, device=dp.device)
        j = torch.full((B,), start_j, dtype=torch.long, device=dp.device)
        for k in range(n + m - 1, -1, -1):
            path[0, k] = i
            path[1, k] = j
            iv = i > 0
            jv = j > 0
            ijv = torch.logical_and(iv, jv)
            ij_lte_i = torch.zeros_like(iv)
            ij_lte_j = torch.zeros_like(jv)
            i_lteq_j = torch.zeros_like(iv)
            ij_lte_i[ijv] = dp[ijv, i[ijv] - 1, j[ijv] - 1] <= dp[ijv, i[ijv] - 1, j[ijv] - 0]
            ij_lte_j[ijv] = dp[ijv, i[ijv] - 1, j[ijv] - 1] <= dp[ijv, i[ijv] - 0, j[ijv] - 1]
            i_lteq_j[ijv] = dp[ijv, i[ijv] - 1, j[ijv] - 0] <= dp[ijv, i[ijv] - 0, j[ijv] - 1]
            ij_is_min = torch.logical_and(ij_lte_i, ij_lte_j)
            i = torch.where(torch.logical_and(iv, torch.logical_or(ij_is_min, i_lteq_j)), i - 1, i)
            j = torch.where(torch.logical_and(jv, torch.logical_or(ij_is_min, ~i_lteq_j)), j - 1, j)
        # make sure indices are not repeated
        rep_mask = (path == path.roll(-1, dims=1)).all(dim=0)
        rep_mask[-1] = False
        path[:, rep_mask] = -1  # mark invalid
        # add batch indices
        batches = torch.arange(B, device=path.device, dtype=path.dtype).expand(1, n + m, B)
        path = torch.cat([batches, path], dim=0)
        # filter invalid entries
        path_v = torch.logical_and(path[1] >= 0, path[2] >= 0)  # valid path
        idx = path[:, path_v]  # [3, K]
        # construct sparse tensor
        val = torch.ones(idx.size(-1), dtype=torch.float32, device=idx.device)
        coupling = torch.sparse_coo_tensor(idx, val, dp.size()).coalesce()
    return coupling


def check_input(a, b, M):
    B, n, m = M.size()
    device = M.device
    dtype = M.dtype
    for x, name in zip([a, b], ["a", "b"]):
        assert not torch.isnan(x).any(), f"{name} must not contain NaN"
        assert not torch.isinf(x).any(), f"{name} must not contain Inf"
        assert x.device == device, f"{name} must be on the same device as M ({device})"
        assert x.dtype == dtype, f"{name} must have the same dtype as M ({dtype})"
    assert a.size() == (B, n), f"a must have shape (B, N)={(B, n)} but got {a.size()}"
    assert b.size() == (B, m), f"b must have shape (B, M)={(B, m)} but got {b.size()}"

    diag_mask = torch.zeros(n, m, dtype=torch.bool, device=device).fill_diagonal_(1)
    assert torch.all(M[:, diag_mask] >= 0.0), "diag(M) must be non-negative"
    assert torch.all(M[:, ~diag_mask] > 0.0), "off-diag(M) must be positive"

    assert torch.allclose(a.sum(dim=-1), b.sum(dim=-1), atol=1e-3, rtol=1e-3), "a and b must have the same sum"

    assert (a > 0.0).all(), "a must be strictly positive"
    assert (b > 0.0).all(), "b must be strictly positive"


if __name__ == "__main__":
    B, n, m = 1, 10, 10
    a = torch.ones(B, n) / n
    b = torch.ones(B, m) / m
    M = torch.rand(B, n, m) + 1e-6

    sinkhorn_log(a, b, M, torch.ones(B, n), torch.ones(B, m), 1.0, 100, 1e-3)

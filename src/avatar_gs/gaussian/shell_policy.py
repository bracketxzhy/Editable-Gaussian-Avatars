from __future__ import annotations

import torch


def estimate_inner_count(
    n_verts: int,
    n_faces: int,
    *,
    k_v: float = 1.2,
    k_f: float = 0.35,
    n_min: int = 80000,
    n_max: int = 220000,
) -> int:
    est = float(k_v) * float(max(int(n_verts), 0)) + float(k_f) * float(max(int(n_faces), 0))
    return int(max(min(int(round(est)), int(n_max)), int(n_min)))


def estimate_outer_budget(
    n_inner: int,
    *,
    init_ratio: float = 0.15,
    max_ratio: float = 0.40,
) -> tuple[int, int]:
    n = max(int(n_inner), 0)
    n_init = int(round(float(init_ratio) * float(n)))
    n_max = int(round(float(max_ratio) * float(n)))
    return max(n_init, 0), max(n_max, 0)


def compute_shell_score(
    *,
    residual_ema: torch.Tensor,
    edge: torch.Tensor,
    curv_conf: torch.Tensor,
    leak: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.7,
    delta: float = 0.6,
) -> torch.Tensor:
    r = residual_ema.detach().float().clamp(min=0.0)
    e = edge.detach().float().clamp(min=0.0)
    c = curv_conf.detach().float().clamp(min=0.0)
    l = leak.detach().float().clamp(min=0.0)
    return float(alpha) * r + float(beta) * e + float(gamma) * c + float(delta) * l


def select_outer_spawn_candidates(
    *,
    shell_score: torch.Tensor,
    visibility: torch.Tensor,
    vis_count: torch.Tensor,
    V_min: int,
    tau_vis: float,
    top_percent: float,
    max_new: int,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    score = shell_score.detach().float()
    vis = visibility.detach().float().clamp(0.0, 1.0)
    vc = vis_count.detach().float()

    eligible = (vc >= float(max(int(V_min), 0))) & (vis >= float(tau_vis))
    if active_mask is not None:
        eligible = eligible & active_mask.detach().bool()

    idx = torch.nonzero(eligible, as_tuple=False).view(-1)
    if int(idx.numel()) == 0:
        return idx

    p = float(max(min(top_percent, 100.0), 0.0))
    if p <= 0.0:
        return torch.empty((0,), dtype=torch.long, device=idx.device)

    k = max(int(round((p / 100.0) * int(idx.numel()))), 1)
    k = min(k, int(max(int(max_new), 0))) if int(max_new) > 0 else k
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=idx.device)
    if k >= int(idx.numel()):
        return idx

    topk = torch.topk(score.index_select(0, idx), k=k, largest=True).indices
    return idx.index_select(0, topk)


def select_inner_softfreeze_candidates(
    *,
    inner_mask: torch.Tensor,
    inner_vis: torch.Tensor,
    inner_contrib: torch.Tensor,
    outer_cover_vis: torch.Tensor,
    outer_cover_contrib: torch.Tensor,
    soft_vis_th: float,
    soft_contrib_th: float,
    outer_vis_th: float,
    outer_contrib_th: float,
) -> torch.Tensor:
    m = inner_mask.detach().bool()
    cond = (
        (inner_vis.detach().float() < float(soft_vis_th))
        & (inner_contrib.detach().float() < float(soft_contrib_th))
        & (outer_cover_vis.detach().float() > float(outer_vis_th))
        & (outer_cover_contrib.detach().float() > float(outer_contrib_th))
    )
    return m & cond


def select_inner_hardprune_candidates(
    *,
    inner_mask: torch.Tensor,
    soft_counter: torch.Tensor,
    hard_counter: torch.Tensor,
    soft_patience: int,
    hard_patience: int,
) -> torch.Tensor:
    m = inner_mask.detach().bool()
    return m & (soft_counter >= float(max(int(soft_patience), 0))) & (hard_counter >= float(max(int(hard_patience), 0)))


def select_outer_prune_candidates(
    *,
    outer_mask: torch.Tensor,
    outer_vis: torch.Tensor,
    outer_score: torch.Tensor,
    low_vis_th: float = 0.02,
    low_score_percent: float = 10.0,
) -> torch.Tensor:
    m = outer_mask.detach().bool()
    vis = outer_vis.detach().float()
    score = outer_score.detach().float()
    idx = torch.nonzero(m & (vis <= float(low_vis_th)), as_tuple=False).view(-1)
    if int(idx.numel()) == 0:
        return idx

    p = float(max(min(low_score_percent, 100.0), 0.0))
    if p <= 0.0:
        return torch.empty((0,), dtype=torch.long, device=idx.device)

    k = max(int(round((p / 100.0) * int(idx.numel()))), 1)
    if k >= int(idx.numel()):
        return idx

    vals = score.index_select(0, idx)
    lowk = torch.topk(vals, k=k, largest=False).indices
    return idx.index_select(0, lowk)

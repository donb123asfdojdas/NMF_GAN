import torch


def effective_mask(mask: torch.Tensor, valid_col_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_col_mask is None:
        return mask
    return mask * valid_col_mask


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid_col_mask: torch.Tensor | None,
) -> torch.Tensor:
    eff_mask = effective_mask(mask, valid_col_mask)
    return (((pred - target) ** 2) * eff_mask).sum() / (eff_mask.sum() + 1e-8)


def observed_consistency_loss(
    pred: torch.Tensor,
    coarse: torch.Tensor,
    mask: torch.Tensor,
    valid_col_mask: torch.Tensor | None,
) -> torch.Tensor:
    observed = (1.0 - mask)
    if valid_col_mask is not None:
        observed = observed * valid_col_mask
    return (((pred - coarse) ** 2) * observed).sum() / (observed.sum() + 1e-8)


def correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid_col_mask: torch.Tensor | None,
) -> torch.Tensor:
    eff_mask = effective_mask(mask, valid_col_mask)
    pred_flat = (pred * eff_mask).flatten(1)
    target_flat = (target * eff_mask).flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((pred_centered.square().sum(dim=1) + 1e-8) * (target_centered.square().sum(dim=1) + 1e-8))
    corr = numerator / denominator
    return (1.0 - corr).mean()

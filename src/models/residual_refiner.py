import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TinyMaskedResidualRefiner(nn.Module):
    def __init__(self, hidden_channels: int = 32, use_valid_mask_feature: bool = True) -> None:
        super().__init__()
        in_channels = 6 + int(use_valid_mask_feature)
        self.use_valid_mask_feature = use_valid_mask_feature
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_channels),
            nn.SiLU(),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def build_features(
        self,
        coarse: torch.Tensor,
        mask: torch.Tensor,
        valid_col_mask: torch.Tensor | None,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
    ) -> torch.Tensor:
        features = [
            coarse,
            mask,
            coarse * mask,
            coarse * (1.0 - mask),
            row_pos,
            col_pos,
        ]
        if self.use_valid_mask_feature:
            valid = valid_col_mask if valid_col_mask is not None else torch.ones_like(coarse)
            features.append(valid)
        return torch.cat(features, dim=1)

    def forward(
        self,
        L: torch.Tensor,
        M: torch.Tensor,
        valid_col_mask: torch.Tensor | None = None,
        row_pos: torch.Tensor | None = None,
        col_pos: torch.Tensor | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if row_pos is None:
            row_pos = torch.linspace(0.0, 1.0, L.shape[-2], device=L.device, dtype=L.dtype).view(1, 1, -1, 1).expand_as(L)
        if col_pos is None:
            col_pos = torch.linspace(0.0, 1.0, L.shape[-1], device=L.device, dtype=L.dtype).view(1, 1, 1, -1).expand_as(L)
        features = self.build_features(L, M, valid_col_mask, row_pos, col_pos)
        residual = self.net(features)
        pred = L + M * residual
        pred = pred * M + L * (1.0 - M)
        if valid_col_mask is not None:
            pred = pred * valid_col_mask + L * (1.0 - valid_col_mask)
            residual = residual * valid_col_mask
        if return_residual:
            return pred, residual
        return pred

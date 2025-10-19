import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, Sequence

from .core import (
    Lambda,
    Patch,
    EnergyLayerNorm,
    PositionEncode,
    Hopfield,
    Attention,
    ETBlock,
)

TENSOR = torch.Tensor


class ET(nn.Module):
    """
    Energy Transformer with trainable per-block step sizes (alpha).

    This mirrors image_et.core.ET but replaces the scalar step size with
    a learned parameter per block, used during the gradient flow update.
    """

    def __init__(
        self,
        x: TENSOR,
        patch: Union[nn.Module, Callable],
        out_dim: Optional[int] = None,
        tkn_dim: int = 256,
        qk_dim: int = 64,
        nheads: int = 12,
        hn_mult: float = 4.0,
        attn_beta: Optional[float] = None,
        attn_bias: bool = False,
        hn_bias: bool = False,
        hn_fn: Callable = lambda x: -0.5 * (F.relu(x) ** 2.0).sum(),
        time_steps: int = 1,
        blocks: int = 1,
        alpha_init: float = 1.0,
        alpha_activation: Optional[str] = "softplus",
    ):
        super().__init__()

        x = patch(x)
        _, n, d = x.shape

        self.K = time_steps
        self.patch = patch

        self.encode = nn.Sequential(
            nn.Linear(d, tkn_dim),
        )

        self.decode = nn.Sequential(
            nn.LayerNorm(tkn_dim, tkn_dim),
            nn.Linear(tkn_dim, out_dim if out_dim is not None else d),
        )

        self.pos = PositionEncode(tkn_dim, n + 1)
        self.cls = nn.Parameter(torch.ones(1, 1, tkn_dim))
        self.mask = nn.Parameter(torch.normal(0, 0.002, size=[1, 1, tkn_dim]))

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        EnergyLayerNorm(tkn_dim),
                        ETBlock(
                            tkn_dim,
                            qk_dim,
                            nheads,
                            hn_mult,
                            attn_beta,
                            attn_bias,
                            hn_bias,
                            hn_fn,
                        ),
                    ]
                )
                for _ in range(blocks)
            ]
        )

        # Trainable per-block alpha parameters (unconstrained). We map them
        # with softplus (default) at use time to ensure positivity.
        init = float(alpha_init)
        # inverse softplus for stable initialization if using softplus
        if alpha_activation == "softplus":
            init_param = torch.log(torch.exp(torch.tensor(init)) - 1.0) if init > 0 else torch.tensor(init)
        else:
            init_param = torch.tensor(init)
        self.alpha_raw = nn.ParameterList([nn.Parameter(init_param.clone()) for _ in range(blocks)])
        self.alpha_activation = alpha_activation

    def _alpha_value(self, b: int, override: Optional[float] = None) -> TENSOR:
        if override is not None:
            return torch.as_tensor(override, dtype=self.alpha_raw[b].dtype, device=self.alpha_raw[b].device)
        if self.alpha_activation == "softplus":
            return F.softplus(self.alpha_raw[b])
        return self.alpha_raw[b]

    def visualize(
        self,
        x: TENSOR,
        mask: Optional[TENSOR] = None,
        alpha: Optional[float] = None,
        *,
        attn_mask: Optional[Sequence[TENSOR]] = None,
    ):
        x = self.patch(x)
        x = self.encode(x)

        if mask is not None:
            x[:, mask] = self.mask

        grad_mask = None
        if mask is not None:
            false_col = torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
            grad_mask = torch.cat([false_col, mask], dim=1)

        x = torch.cat([self.cls.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pos(x)

        energies = []
        embeddings = [self.patch(self.decode(x)[:, 1:], reverse=True)]

        for b, (norm, et) in enumerate(self.blocks):
            a = self._alpha_value(b, override=alpha)
            for _ in range(self.K):
                g = norm(x)
                dEdg, E = torch.func.grad_and_value(et)(g, attn_mask)
                if grad_mask is not None:
                    dEdg = dEdg.masked_fill(~grad_mask[..., None], 0.0)
                x = x - a * dEdg
                energies.append(E)
                embeddings.append(self.patch(self.decode(x)[:, 1:], reverse=True))

        g = norm(x)
        energies.append(et(g, attn_mask))
        return energies, embeddings

    def evolve(
        self,
        x: TENSOR,
        alpha: Optional[float],
        mask: Optional[TENSOR] = None,
        *,
        attn_mask: Optional[Sequence[TENSOR]] = None,
        return_energy: bool = False,
    ):
        energies = [] if return_energy else None

        for b, (norm, et) in enumerate(self.blocks):
            a = self._alpha_value(b, override=alpha)
            for _ in range(self.K):
                g = norm(x)
                dEdg, E = torch.func.grad_and_value(et)(g, attn_mask)

                x = x - a * dEdg
                if return_energy:
                    energies.append(E)

        if return_energy:
            g = norm(x)
            E = et(g, attn_mask)
            energies.append(E)

        return x, energies

    def forward(
        self,
        x: TENSOR,
        mask: Optional[TENSOR] = None,
        attn_mask: Optional[Sequence[TENSOR]] = None,
        *,
        alpha: Optional[float] = None,
        return_energy: bool = False,
        use_cls: bool = False,
    ):
        x = self.patch(x)
        x = self.encode(x)

        if mask is not None:
            x[mask] = self.mask

        x = torch.cat([self.cls.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pos(x)

        x, energies = self.evolve(
            x, alpha, mask=None, attn_mask=attn_mask, return_energy=return_energy
        )

        x = self.decode(x)
        yh = x[:, :1] if use_cls else x[:, 1:]

        if return_energy:
            return yh, energies
        return yh


# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, List, Tuple, Generator
from functools import lru_cache

from tqdm import trange

import torch
from torch import roll, Tensor

from .mobility import MOBILITY_TYPE


def integrate_step(
        u: Tensor,
        h: float,
        dim: int,
        eps: float,
        u_gel: float,
        mobility_func: MOBILITY_TYPE,
) -> Tensor:

    roll_axes: List[Tuple[int, int]] = _get_axes(dim)

    grids: List[Tensor] = [roll(u, *ax) for ax in roll_axes]

    ddu = sum(grids) - 2 * dim * u

    mob_grids: List[Tensor] = [mobility_func(0.5 * (u + p), u_gel) for p in grids]

    mu_mtx: Tensor = - eps * u + u ** 3 - 1 / h ** 2 * ddu

    res: Tensor = 1 / h ** 2 * sum(m * (roll(mu_mtx, *ax) - mu_mtx) for m, ax in zip(mob_grids, roll_axes))

    return res


## legacy code for a better understanding of the function above (3d)
def _integrate_step3d(
        u: Tensor,
        h: float,
        eps: Union[float, Tensor],
        u_gel: Union[float, Tensor],
        mobility_func: MOBILITY_TYPE,
) -> Tensor:
    ip = roll(u, 1, -1)
    im = roll(u, -1, -1)
    jp = roll(u, 1, -2)
    jm = roll(u, -1, -2)
    km = roll(u, -1, -3)
    kp = roll(u, 1, -3)

    ddu = ip + im + jp + jm + km + kp - 6 * u

    mip, mim, mjp, mjm, mkm, mkp = (
        mobility_func(0.5 * (u + ip), u_gel),
        mobility_func(0.5 * (u + im), u_gel),
        mobility_func(0.5 * (u + jp), u_gel),
        mobility_func(0.5 * (u + jm), u_gel),
        mobility_func(0.5 * (u + km), u_gel),
        mobility_func(0.5 * (u + kp), u_gel),
    )

    mu_mtx = - eps * u + u ** 3 - 1 / h ** 2 * ddu

    return 1 / h ** 2 * (
            mip * (roll(mu_mtx, 1, -1) - mu_mtx) +
            mim * (roll(mu_mtx, -1, -1) - mu_mtx) +
            mjp * (roll(mu_mtx, 1, -2) - mu_mtx) +
            mjm * (roll(mu_mtx, -1, -2) - mu_mtx) +
            mkp * (roll(mu_mtx, 1, -3) - mu_mtx) +
            mkm * (roll(mu_mtx, -1, -3) - mu_mtx)
    )


@lru_cache()
def _get_axes(dim: int) -> List[Tuple[int, int]]:
    return [(-1 if d % 2 else 1, - (d // 2) - 1) for d in range(2 * dim)]


def solver_gen(
        u: Tensor,
        h: float,
        tau: float,
        dim: int,
        eps: float,
        u_gel: float,
        mobility_func: MOBILITY_TYPE,
        time_steps: int,
        time_delta: int = 1,
        disable_tqdm: bool = False,
) -> Generator[Tensor, None, None]:
    for t in trange(1, time_steps, disable=disable_tqdm):
        u = u + tau * integrate_step(u, h, dim, eps, u_gel, mobility_func)
        if not t % time_delta:
            yield u


def get_init_u(
        grid_shape: Tuple[int, ...],
        eps: float,
        noise_amp: float,
        mean_u: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cuda'),
) -> Tensor:
    amp = noise_amp * (1 - eps)
    noise = 2 * torch.rand(*grid_shape, device=device, dtype=dtype) - 1
    return mean_u + noise * amp


def solver(
        grid_shape: Tuple[int, ...],
        h: float,
        tau: float,
        eps: float,
        u_gel: float,
        noise_amp: float,
        mean_u: float,
        mobility_func: MOBILITY_TYPE,
        time_steps: int,
        time_delta: int = 1,
        disable_tqdm: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cuda'),
) -> Tensor:
    dim = len(grid_shape)
    time_size = time_steps // time_delta

    init_u = get_init_u(grid_shape, eps, noise_amp, mean_u, dtype=dtype, device=device)
    res = torch.empty(time_size, *grid_shape, dtype=dtype, device=device)
    res[0] = init_u

    for t, u in enumerate(
            solver_gen(init_u, h, tau, dim, eps, u_gel, mobility_func, time_steps, time_delta, disable_tqdm),
            start=1
    ):
        res[t] = u

    return res

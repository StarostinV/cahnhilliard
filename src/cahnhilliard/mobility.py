# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from torch import Tensor, exp

MOBILITY_TYPE = Callable[[Tensor, float], Tensor]


def get_mobility_func(alpha: float) -> MOBILITY_TYPE:
    def mobility_func(u: Tensor, u_gel: float) -> Tensor:
        return 1. / (1. + exp(alpha * (u - u_gel)))

    return mobility_func

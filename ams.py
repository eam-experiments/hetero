# Copyright [2023] Luis Alberto Pineda Cortés & Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import constants
from associative import AssociativeMemory
from hetero_associative import HeteroAssociativeMemory


class AssociativeMemorySystem:
    """ Associative Memory System.

    Includes two homo-associative memory registers, and an
    hetero-associative register to associate their content.
    """

    def __init__(self, n: int, m: int, p: int, q: int,
            left_xi = constants.xi_default,
            left_iota = constants.iota_default,
            left_kappa = constants.kappa_default,
            left_sigma = constants.sigma_default,
            mid_xi = constants.xi_default,
            mid_iota = constants.iota_default,
            mid_kappa = constants.kappa_default,
            mid_sigma = constants.sigma_default,
            right_xi = constants.xi_default,
            right_iota = constants.iota_default,
            right_kappa = constants.kappa_default,
            right_sigma = constants.sigma_default):
        self.left_mem = AssociativeMemory(n, m,
                xi = left_xi, iota = left_iota,
                kappa = left_kappa, sigma = left_sigma)
        self.right_mem = AssociativeMemory(p, q,
                xi = right_xi, iota = right_iota,
                kappa = right_kappa, sigma = right_sigma)
        self.heter_mem = HeteroAssociativeMemory(n, p, m, q,
                xi = mid_xi, iota = mid_iota,
                kappa = mid_kappa, sigma = mid_sigma)
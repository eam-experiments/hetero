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

import numpy as np
import constants
from associative import AssociativeMemory
from hetero_associative import HeteroAssociativeMemory


class AssociativeMemorySystem:
    """ Associative Memory System.

    Includes two homo-associative memory registers, and an
    hetero-associative register to associate their content.
    """

    def __init__(self, n: int, p: int, m: int, q: int,
            xi = constants.xi_default,
            iota = constants.iota_default,
            kappa = constants.kappa_default,
            sigma = constants.sigma_default):
        self.left_mem = AssociativeMemory(n, m,
                xi = xi, iota = iota,
                kappa = kappa, sigma = sigma)
        self.right_mem = AssociativeMemory(p, q,
                xi = xi, iota = iota,
                kappa = kappa, sigma = sigma)
        self.heter_mem = HeteroAssociativeMemory(n, p, m, q,
                xi = xi, iota = iota,
                kappa = kappa, sigma = sigma)
        
    @property
    def entropy(self):
        return self.heter_mem.entropy

    def register(self, vector_a_p, vector_b_p):
        self.left_mem.register(vector_a_p)
        self.right_mem.register(vector_b_p)
        vector_a, recognized_a, _ = self.left_mem.recall(vector_a_p)
        vector_b, recognized_b, _ = self.right_mem.recall(vector_b_p)
        if recognized_a and recognized_b:
            self.heter_mem.register(vector_a, vector_b)
            return True
        else:
            return False

    def recognize(self, vector_a_p, vector_b_p):
        vector_a, recognized, weights_a = self.left_mem.recall_detailed_weights(vector_a_p)
        if not recognized:
            return False, 0
        vector_b, recognized, weights_b = self.right_mem.recall_detailed_weights(vector_b_p)
        if not recognized:
            return False, 0
        recognized, weight = self.heter_mem.recog_weighted(
            vector_a, vector_b, weights_a, weights_b)
        return recognized, weight

    def recall_from_left(self, vector_a_p):
        vector_a, recognized, weights_a = self.left_mem.recall_detailed_weights(vector_a_p)
        if not recognized:
            return self.right_mem.undefined_output, recognized, 0
        weight_a = np.mean(weights_a)
        vector_b, weight_h = self.heter_mem.recall_from_left_weighted(vector_a, weights_a) 
        vector_b_p, recognized, weight_b = self.right_mem.recall(vector_b)
        return vector_b_p, recognized, ((weight_a + weight_h + weight_b)/3 
                                        if recognized else 0)
        
    def recall_from_right(self, vector_b_p):
        vector_b, recognized, weights_b = self.right_mem.recall_detailed_weights(vector_b_p)
        if not recognized:
            return self.left_mem.undefined_output, recognized, 0
        vector_a, weight = self.heter_mem.recall_from_right_weighted(vector_b, weights_b) 
        vector_a_p, recognized, _ = self.left_mem.recall(vector_a)
        return vector_a_p, recognized, (weight if recognized else 0)


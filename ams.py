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
            left_es: constants.ExperimentSettings,
            right_es: constants.ExperimentSettings,
            hetero_es: constants.ExperimentSettings):
        self.left_mem = AssociativeMemory(n, m, left_es)
        self.right_mem = AssociativeMemory(p, q, right_es)
        self.heter_mem = HeteroAssociativeMemory(n, p, m, q,
                hetero_es)
        
    @property
    def entropy(self):
        return self.heter_mem.entropy

    def register(self, vector_a, vector_b):
        self.left_mem.register(vector_a)
        self.right_mem.register(vector_b)
        self.heter_mem.register(vector_a, vector_b)

    def recognize_left(self, vector_a):
        recognized, weight = self.left_mem.recognize(vector_a)
        return recognized, weight

    def recognize_right(self, vector_b):
        recognized, weight = self.right_mem.recognize(vector_b)
        return recognized, weight

    def recognize_heter(self, vector_a, vector_b):
        recognized, weight = self.heter_mem.recognize(
            vector_a, vector_b)
        return recognized, weight

    def recall_from_left(self, vector_a):
        vector_b, weight = self.heter_mem.recall_from_left(vector_a)
        recognized = (np.count_nonzero(np.isnan(vector_b)) != vector_b.size)
        return vector_b, recognized, weight
        
    def recall_from_right(self, vector_b):
        vector_a, weight = self.heter_mem.recall_from_right(vector_b)
        recognized = (np.count_nonzero(np.isnan(vector_a)) != vector_a.size)
        return vector_a, recognized, weight


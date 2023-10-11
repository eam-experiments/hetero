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

import math
import random
import string
from joblib import Parallel, delayed
from multiprocessing import shared_memory
import numpy as np

import constants

class HeteroAssociativeMemory3D:
    def __init__(self, n: int, p: int, m: int, q: int,
        es: constants.ExperimentSettings):
        """
        Parameters
        ----------
        n : int
            The size of the first domain (of properties).
        m : int
            The size of the first range (of representation).
        p : int
            The size of the second domain (of properties).
        q : int
            The size of the second range (of representation).
        es: Experimental Settings
            Includes the values for iota, kappa, xi y sigma.
        """
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.absolute_max = 2**32 - 1
        self.xi = es.xi
        self.sigma = es.sigma
        self.iota = es.iota
        self.kappa = es.kappa
        self._top = self.m + self.q
        self.relation = np.zeros((self.n, self.p, self._top, self.n_vars), dtype=int)
        self._iota_relation = np.zeros((self.n, self.p, self._top, self.n_vars), dtype=int)
        self._entropies = np.zeros((self.n, self.p), dtype=np.double)
        self._means = np.zeros((self.n, self.p), dtype=np.double)
        self._updated = True
        print(f'Relational memory {{n: {self.n}, p: {self.p}, ' +
            f'm: {self.m}, q: {self.q}, z: {self._top}, ' +
            f'xi: {self.xi}, iota: {self.iota}, ' +
            f'kappa: {self.kappa}, sigma: {self.sigma}}}, has been created')


    def __str__(self):
        return f'{{n: {self.n}, p: {self.p}, m: {self.m}, q: {self.q},\n{self.rel_string}}}'

    @property
    def model_name(self):
        return constants.d3_model_name
    
    @property
    def n_vars(self):
        return 3
    
    @property
    def w_index(self):
        return 0
    
    @property
    def v_index(self):
        """Index of value in the projection."""
        return 1
    
    @property
    def a_index(self):
        return 1
    
    @property
    def b_index(self):
        return 2
    
    @property
    def undefined(self):
        return self._top
    
    def is_undefined(self, value):
        return value == self.undefined
    
    @property
    def entropies(self):
        if not self._updated:
            self._updated = self.update()
        return self._entropies

    @property
    def entropy(self):
        """Return the entropy of the Hetero Associative Memory."""
        return np.mean(self.entropies)

    @property
    def means(self):
        if not self._updated:
            self._updated = self.update()
        return self._means

    @property
    def mean(self):
        """Return the mean of the weights in the Hetero Associative Memory"""
        return np.mean(self.means)

    @property
    def iota_relation(self):
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation

    @property
    def fullness(self):
        count = np.count_nonzero(self.relation[:, :, :, self.w_index])
        total = self.n*self.p*self._top
        return count*1.0/total
    
    @property
    def rel_string(self):
        return self._to_string(self.relation)

    def alt(self, dim):
        return (dim + 1) % 2

    def cols(self, dim):
        return self.n if dim == 0 else self.p

    def rows(self, dim):
        return self.m if dim == 0 else self.q

    def register(self, cue_a, cue_b, weights_a = None, weights_b = None):
        if weights_a is None:
            weights_a = np.full(len(cue_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(cue_b), fill_value=1)
        cue_a = self.validate(cue_a, 0)
        cue_b = self.validate(cue_b, 1)
        r_io = self.vectors_to_relation(cue_a, cue_b, weights_a, weights_b)
        self.abstract(r_io)

    def recognize(self, cue_a, cue_b, weights_a = None, weights_b = None):
        if weights_a is None:
            weights_a = np.full(len(cue_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(cue_b), fill_value=1)
        recognized, weights = self._recog(cue_a, cue_b, weights_a, weights_b, final = False)
        weight = np.mean(weights)
        recognized = recognized and (self.kappa*self.mean <= weight)
        return recognized, weight

    def _recog(self, cue_a, cue_b, weights_a, weights_b, final = True):
        cue_a = self.validate(cue_a, 0)
        cue_b = self.validate(cue_b, 1)
        r_io = self.vectors_to_relation(cue_a, cue_b, weights_a, weights_b)
        implication = self.containment(r_io)
        recognized = np.count_nonzero(implication == 0) <= self.xi
        weights = self._weights(r_io)
        if final:
            recognized = recognized and (self.kappa*self.mean <= np.mean(weights))
        return recognized, weights

    def recall_from_left(self, cue, weights = None):
        if weights is None:
            weights = np.full(len(cue), fill_value=1)
        return self._recall(cue, weights, 0)

    def recall_from_right(self, cue, weights = None):
        if weights is None:
            weights = np.full(len(cue), fill_value=1)
        return self._recall(cue, weights, 1)

    def _recall(self, cue, weights, dim):
        cue = self.validate(cue, dim)
        projection = self.project(cue, weights, dim)
        projection = self.transform(projection)
        projection_weights = np.sum(projection[:,:,self.w_index], axis=1)
        recognized = (np.count_nonzero(projection_weights == 0) <= self.xi)
        r_io, r_io_weights = self.reduce(projection, self.alt(dim))
        r_io_w = np.mean(r_io_weights)
        recognized = recognized and (self.kappa*self.mean <= r_io_w)
        r_io = self.revalidate(r_io, self.alt(dim))
        return r_io, recognized, r_io_w, projection

    def abstract(self, r_io):
        self.relation[:, :, :, self.w_index] = np.where(
            self.relation[:, :, :, self.w_index] == self.absolute_max,
            self.relation[:, :, :, self.w_index], self.relation[:, :, :, self.w_index] + r_io[:, :, :, self.w_index])
        for i in [self.a_index, self.b_index]:
            self.relation[:, :, :, i] = (self.relation[:, :, :, self.w_index]-r_io[:, :, :, self.w_index])\
                / np.where(self.relation[:, :, :, self.w_index] == 0, 1, self.relation[:, :, :, self.w_index]) \
                    * self.relation[:, :, :, i] \
                        + (r_io[:, :, :, i]*r_io[:, :, :, self.w_index]) / \
                            np.where(self.relation[:, :, :, self.w_index] == 0, 1, self.relation[:, :, :, self.w_index])
        self._updated = False

    def containment(self, r_io):
        return np.where((r_io[:, :, :, self.w_index] == 0) | (self.iota_relation[:, :, :, self.w_index] != 0))

    def project(self, cue, weights, dim):
        projection = np.zeros((self.cols(self.alt(dim)), self._top, self.n_vars), dtype=int)
        chosen = self.filter_relation(cue, weights, dim)
        alt_index = self.b_index if dim == 0 else self.a_index
        for j in range(self.cols(self.alt(dim))):
            for k in range(self._top):
                v = np.sum(chosen[:, j, k, self.b_index] if dim == 0 else chosen[j, :, k, self.a_index])
                w = np.sum(chosen[:, j, k, self.w_index] if dim == 0 else chosen[j, :, k, self.w_index])
                n = np.count_nonzero(chosen[:, j, k, self.w_index] if dim == 0 else chosen[j, :, k, self.w_index])
                projection[j, k, alt_index] = v if n == 0 else int(v/n)
                projection[j, k, self.w_index] = w if n == 0 else int(w/n)
        return projection

    def filter_relation(self, cue, weights, dim):
        chosen = np.zeros(self.relation.shape, dtype=int)
        the_index = self.a_index if dim == 0 else self.b_index
        alt_index = self.b_index if dim == 0 else self.a_index
        for i in range(self.cols(dim)):
            value = cue[i]
            for j in range(self.cols(self.alt(dim))):
                a = i if dim == 0 else j
                b = j if dim == 0 else i
                distance = float('inf')
                weight = 0
                index = self._top
                for k in range(self._top):
                    v = self.relation[a, b, k, the_index]
                    w = self.relation[a, b, k, self.w_index]
                    if (abs(value - v) < distance) or ((abs(value - v) == distance) and (weight < w)):
                        index = k
                        distance = abs(value - v)
                        weight = w
                chosen[a, b, index, self.w_index] = weight
                chosen[a, b, index, the_index] = value
                chosen[a, b, index, alt_index] = self.relation[a, b, index, alt_index]
        return chosen

    # Reduces a relation to a function
    def reduce(self, projection, dim):
        cols = self.cols(dim)
        v = np.array([self.choose(column, dim)
                for column in projection])
        weights = []
        for i in range(cols):
            if self.is_undefined(v[i]):
                weights.append(0)
            else:
                weights.append(projection[i, v[i], self.w_index])
        weights = np.array(weights)
        return v, weights

    
    def choose(self, column, dim):
        """Choose a value from the column.
        
        It assumes the column as a probabilistic distribution.
        """
        dist = column
        s = dist[:, self.w_index].sum()
        if s == 0:
            return random.randrange(dist.size)
        r = s*random.random()
        index = self.a_index if dim == 0 else self.b_index
        for j in range(dist.shape[0]):
            if r <= dist[j, self.w_index]:
                return dist[j, index]
            r -= dist[j, self.w_index]
        return self.undefined

    def _weights(self, r_io):
        r = r_io*np.count_nonzero(r_io)/np.sum(r_io)
        weights = np.sum(r[:, :, :, self.w_index] * self.relation[:, :, :, self.w_index], axis=2)
        return weights
        
    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_entropies(self):
        for i in range(self.n):
            for j in range(self.p):
                relation = self.relation[i, j, :, self.w_index]
                total = np.sum(relation)
                if total > 0:
                    matrix = relation/total
                else:
                    matrix = relation
                matrix = np.multiply(-matrix, np.log2(np.where(matrix == 0.0, 1.0, matrix)))
                self._entropies[i, j] = np.sum(matrix)
        print(f'Entropy updated to mean = {np.mean(self._entropies)}, ' 
              + f'stdev = {np.std(self._entropies)}')

    def _update_means(self):
        for i in range(self.n):
            for j in range(self.p):
                r = self.relation[i, j, :, self.w_index]
                count = np.count_nonzero(r)
                count = 1 if count == 0 else count
                self._means[i,j] = np.sum(r)/count

    def _update_iota_relation(self):
        for i in range(self.n):
            for j in range(self.p):
                matrix = self.relation[i, j, :, self.w_index]
                s = np.sum(matrix)
                if s == 0:
                    self._iota_relation[i, j, :, self.w_index] = np.zeros(self._top, dtype=int)
                else:
                    count = np.count_nonzero(matrix)
                    threshold = self.iota*s/count
                    self._iota_relation[i, j, :, self.w_index] = np.where(matrix < threshold, 0, matrix)
        turned_off = np.count_nonzero(
            self.relation[:, :, :, self.w_index]) - np.count_nonzero(self._iota_relation[:, :, :, self.w_index])
        print(f'Iota relation updated, and {turned_off} cells have been turned off')

    def validate(self, cue, dim):
        """ It asumes vector is an array of floats, and np.nan
            is used to register an undefined value, but it also
            considerers any negative number or out of range number
            as undefined.
        """
        expected_length = self.cols(dim)
        if len(cue.shape) > 1:
            raise ValueError(f'Expected shape ({expected_length},) ' +
                    'but got shape {vector.shape}')
        if cue.size != expected_length:
            raise ValueError('Invalid lenght of the input data. Expected' +
                    f'{expected_length} and given {cue.size}')
        if np.count_nonzero(np.isnan(cue)) > 0:
            raise ValueError('This 3D model does not accept partial functions')
        min = np.min(cue)
        max = np.max(cue)
        if min < 0:
            raise ValueError(f'Value out of range: {min}')
        if max >= self.rows(dim):
            raise ValueError(f'Value out of range: {max}')
        v = np.fix(cue)
        return v.astype('int')

    def revalidate(self, memory, dim):
        v = np.where(memory == self.undefined, np.nan, memory)
        return v

    def vectors_to_relation(self, cue_a, cue_b, weights_a, weights_b):
        relation = np.zeros((self.n, self.p, self._top, self.n_vars), dtype=int)
        for i in range(self.n):
            a = cue_a[i]
            for j in range(self.p):
                b = cue_b[j]
                k = a + b
                w = weights_a[i]*weights_b[j]
                relation[i, j, k, self.w_index] = int(w)
                relation[i, j, k, self.a_index] = a
                relation[i, j, k, self.b_index] = b
        return relation

    def _to_string(self, a, p = ''):
        if a.ndim == 1:
            return f'{p}{a}'
        s = f'{p}[\n'
        for b in a:
            ss = self._to_string(b, p + ' ')
            s = f'{s}{ss}\n'
        s = f'{s}{p}]'
        return s

    def transform(self, r):
        return r if constants.projection_transform == constants.project_same \
            else self.maximum(r) if constants.projection_transform == constants.project_maximum \
            else self.logistic(r)
    
    def maximum(self, r):
        q = np.zeros(r.shape, dtype=float)
        for i in range(r.shape[0]):
            c = r[i]
            L = np.max(c)
            q[i] = np.where(c == L, c, 0.0)
        return q
            
    def logistic(self, r):
        q = np.zeros(r.shape, dtype=float)
        for i in range(r.shape[0]):
            c = r[i]
            L = np.max(c)
            k = 10
            x0 = 0.5
            q[i] = L / (1 + np.exp(-k*(c/L - x0)))
        return q

    def get_random_string(self):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        random_string = ''.join(random.choice(letters) for i in range(constants.random_string_length))
        return random_string

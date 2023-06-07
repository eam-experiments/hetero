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
import time
import random
import numpy as np

import constants

class HeteroAssociativeMemory:
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
        self._n = n
        self._m = m+1 # +1 to handle partial functions.
        self._p = p
        self._q = q+1 # +1 to handle partial functions.
        self._xi = es.xi
        self._absolute_max = 2**16 - 1
        self._sigma = es.sigma
        self._iota = es.iota
        self._kappa = es.kappa
        self._relation = np.zeros((self._n, self._p, self._m, self._q), dtype=np.int)
        self._iota_relation = np.zeros((self._n, self._p, self._m, self._q), dtype=np.int)
        self._entropies = np.zeros((self._n, self._p), dtype=np.double)
        self._means = np.zeros((self._n, self._p), dtype=np.double)
        self._updated = True
        print(f'Relational memory {{n: {self.n}, p: {self.p}, ' +
            f'm: {self.m}, q: {self.q}, ' +
            f'xi: {self.xi}, iota: {self.iota}, ' +
            f'kappa: {self.kappa}, sigma: {self.sigma}}}, has been created')

    def __str__(self):
        return f'{{n: {self.n}, p: {self.p}, m: {self.m}, q: {self.q},\n{self.rel_string}}}'

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return self._p

    @property
    def m(self):
        return self._m-1

    @property
    def q(self):
        return self._q-1

    @property
    def relation(self):
        return self._relation[:, :, :self.m, :self.q]

    @property
    def absolute_max_value(self):
        return self._absolute_max

    @property
    def entropies(self):
        if not self._updated:
            self._updated = self.update()
        return self._entropies

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        return np.mean(self.entropies)

    @property
    def means(self):
        if not self._updated:
            self._updated = self.update()
        return self._means

    @property
    def mean(self):
        return np.mean(self.means)

    @property
    def iota_relation(self):
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation[:, :, :self.m, :self.q]

    @property
    def _full_iota_relation(self):
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if sigma < 0:
            raise ValueError('Sigma must be a non negative number.')
        self._sigma = abs(sigma)

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        if kappa < 0:
            raise ValueError('Kappa must be a non negative number.')
        self._kappa = kappa

    @property
    def iota(self):
        return self._iota

    @iota.setter
    def iota(self, iota):
        if iota < 0:
            raise ValueError('Iota must be a non negative number.')
        self._iota = iota

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, x):
        if x < 0:
            raise ValueError('Xi must be a non negative number.')
        self._xi = x
        self._updated = False

    def undefined(self, dim: int):
        return self.m if dim == 0 else self.q

    def undefined_alt(self, dim: int):
        return self.q if dim == 0 else self.m

    def is_undefined(self, value, dim):
        return value == self.undefined(dim)

    def alt(self, dim):
        return (dim + 1) % 2

    def cols(self, dim):
        return self.n if dim == 0 else self.p

    def cols_alt(self, dim):
        return self.p if dim == 0 else self.n

    def rows(self, dim):
        return self.m if dim == 0 else self.q

    def rows_alt(self, dim):
        return self.q if dim == 0 else self.m

    def core(self, full_relation):
        return full_relation[:, :, :self.m, :self.q]

    def register(self, vector_a, vector_b, weights_a = None, weights_b = None) -> None:
        if weights_a is None:
            weights_a = np.full(len(vector_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(vector_b), fill_value=1)
        vector_a = self.validate(vector_a, 0)
        vector_b = self.validate(vector_b, 1)
        r_io = self.vectors_to_relation(vector_a, vector_b, weights_a, weights_b)
        self.abstract(r_io)

    def recognize(self, vector_a, vector_b, weights_a = None, weights_b = None):
        if weights_a is None:
            weights_a = np.full(len(vector_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(vector_b), fill_value=1)
        recognized, weights = self._recog(vector_a, vector_b, weights_a, weights_b)
        return recognized, np.sum(weights)

    def _recog(self, vector_a, vector_b, weights_a, weights_b):
        vector_a = self.validate(vector_a, 0)
        vector_b = self.validate(vector_b, 1)
        r_io = self.vectors_to_relation(vector_a, vector_b, weights_a, weights_b)
        implication = self.containment(r_io)
        recognized = np.count_nonzero(implication == False) <= self._xi
        weights = self._weights(r_io)
        recognized = recognized and (self._kappa*self.mean <= np.sum(weights))
        return recognized, weights

    def recall_from_left(self, vector):
        return self._recall(vector, 0)

    def recall_from_right(self, vector):
        return self._recall(vector, 1)
    
    def _recall(self, vector, dim):
        vector = self.validate(vector, dim)
        relation = self.project(vector, dim)
        r_io, weight = self.reduce(relation, self.alt(dim))
        recognized = (np.count_nonzero(r_io == self.undefined) == 0)
        r_io = self.revalidate(r_io, self.alt(dim))
        return r_io, recognized, weight, relation

    def abstract(self, r_io):
        self._relation = np.where(
            self._relation == (self.absolute_max_value-1),
            self._relation, self._relation + r_io)
        self._updated = False

    def containment(self, r_io):
        r = r_io[:, :, :self.m, :self.q]
        r_iota = self.iota_relation
        return np.where((r == 0) | (r_iota != 0), True, False)

    def project(self, vector, dim):
        projection = np.zeros((self.cols_alt(dim), self.rows_alt(dim)+1), dtype=int)
        for i in range(self.cols(dim)):
            k = vector[i]
            projection = projection + (self._full_iota_relation[i, :, k, :] if dim == 0
                else self._full_iota_relation[:, i, :, k])
        return projection

    # Reduces a relation to a function
    def reduce(self, relation, dim):
        cols = self.cols(dim)
        v = np.array([self.choose(column, self.cue(column, dim), dim) for column in relation])
        weights = np.array([relation[i, v[i]] for i in range(cols)])
        return v, np.mean(weights)

    # Choose a value from the column, assuming it is a probabilistic distribution.
    def choose(self, column, value, dim):
        dist =  column if value == self.undefined(dim) \
            else self._normalize(column, value, dim)
        s = dist.sum()
        if s == 0:
            return self.undefined(dim)
        n = s*random.random()
        for j in range(dist.size):
            if n < dist[j]:
                return j
            n -= dist[j]
        return self.rows(dim) - 1

    def cue(self, column, dim):
        """ Returns the median of the column.
        """
        s = np.sum(column)
        if s == 0:
            return self.undefined(dim)
        n = s/2.0
        for i, f in enumerate(column):
            if n < f:
                return i
            n -= f

    def _normalize(self, column, cue, dim):
        mean = cue
        rows = self.rows(dim)
        stdv = self.sigma*rows
        scale = 1.0/self.normpdf(0, 0, stdv)
        norm = np.array([self.normpdf(i, mean, stdv, scale) for i in range(rows)])
        return norm*column[:rows]

    def normpdf(self, x, mean, stdv, scale = 1.0):
        var = float(stdv)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return scale*num/denom

    def _weights(self, r_io):
        r = r_io/np.sum(r_io)
        weights = np.sum(r[:, :, :self.m, :self.q] * self.relation, axis=(2,3))
        return weights
        
    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_entropies(self):
        for i in range(self.n):
            for j in range(self.p):
                relation = self.relation[i, j, :, :]
                total = np.sum(relation)
                matrix = relation/total
                matrix = np.multiply(-matrix, np.log2(np.where(matrix == 0.0, 1.0, matrix)))
                self._entropies[i, j] = np.sum(matrix)

    def _update_means(self):
        for i in range(self.n):
            for j in range(self.p):
                r = self.relation[i, j, :, :]
                count = np.count_nonzero(r)
                count = 1 if count == 0 else count
                self._means[i,j] = np.sum(r)/count

    def _update_iota_relation(self):
        for i in range(self.n):
            for j in range(self.p):
                matrix = self.relation[i, j, :, :]
                s = np.sum(matrix)
                if s == 0:
                    self._iota_relation[i, j, :, :] = np.zeros((self._m, self._q), dtype=int)
                else:
                    count = np.count_nonzero(matrix)
                    threshold = self.iota*s/count
                    self._iota_relation[i, j, :self.m, :self.q] = \
                        np.where(matrix < threshold, 0, matrix)
        turned_off = np.count_nonzero(self._relation) - np.count_nonzero(self._iota_relation)
        print(f'Iota relation updated, and {turned_off} cells have been turned off')

    def validate(self, vector, dim):
        """ It asumes vector is an array of floats, and np.nan
            is used to register an undefined value, but it also
            considerers any negative number or out of range number
            as undefined.
        """
        expected_length = self.cols(dim)
        if vector.size != expected_length:
            raise ValueError('Invalid lenght of the input data. Expected' +
                 f'{expected_length} and given {vector.size}')
        undefined = self.undefined(dim)
        v = np.nan_to_num(vector, copy=True, nan=undefined)
        v = np.where((v < 0) | (undefined < v), undefined, v)
        v = v.round()
        return v.astype('int')

    def revalidate(self, vector, dim):
        v = np.where(vector == self.undefined(dim), np.nan, vector)
        return v

    def vectors_to_relation(self, vector_a, vector_b, weights_a, weights_b):
        relation = np.zeros((self._n, self._p, self._m, self._q), dtype=float)
        for i in range(self.n):
            k = vector_a[i]
            for j in range(self.p):
                l = vector_b[j]
                relation[i, j, k, l] = weights_a[i]*weights_b[j]
        return relation

    @property
    def fullness(self):
        count = np.count_nonzero(self.relation)
        total = self.n*self.m*self.p*self.q
        return count*1.0/total
    
    @property
    def rel_string(self):
        return self.toString(self.relation)

    def toString(self, a, p = ''):
        if a.ndim == 1:
            return f'{p}{a}'
        s = f'{p}[\n'
        for b in a:
            ss = self.toString(b, p + ' ')
            s = f'{s}{ss}\n'
        s = f'{s}{p}]'
        return s

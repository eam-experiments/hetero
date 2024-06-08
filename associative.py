# Copyright [2020-23] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
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

# File originally create by Raul Peralta-Lozada.

import math
import random
import numpy as np

import commons

def normpdf(x, mean, sd, scale = 1.0):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return scale*num/denom


class AssociativeMemory:
    def __init__(self, n: int, m: int, es: commons.ExperimentSettings = None, relation = None):
        """
        Parameters
        ----------
        n : int
            The size of the domain (of properties).
        m : int
            The size of the range (of representation).
        es: Experimental Settings, that include xi: The number of
            mismatches allowed between the memory content and the cue.
            iota: Proportion of the mean weight per column a cue must
            have to match the memory (moderated by xi). kappa:
            Proportion of the average mean weight a cue must have to
            match the memory. sigma: The standard deviation of the
            normal distribution used in remembering, as percentage of
            the number of characteristics.
        """
        self._n = n
        self._m = m+1
        if es is None:
            es = commons.ExperimentSettings()
        self._xi = es.xi
        self._sigma = es.sigma
        self._sigma_scaled = es.sigma*m
        self._iota = es.iota
        self._kappa = es.kappa
        self._scale = 1.0/normpdf(0, 0, self._sigma_scaled)
        self._absolute_max = 2**16 - 1

        # It is m+1 to handle partial functions.
        self._relation = np.zeros((self._n, self._m), dtype=int)
        if relation is not None:
            self._relation[:, :m] = relation
        # Iota moderated relation
        self._iota_relation = np.zeros((self._n, self._m), dtype=int)
        self._entropies = np.zeros(self._n, dtype=float)
        self._means = np.zeros(self._n, dtype=float)

        # A flag to know whether iota-relation, entropies and means
        # are up to date.
        self._updated = True if relation is None else self.update()
        print(f'Memory {{n: {self.n}, m: {self.m}, ' +
            f'xi: {self.xi}, iota: {self.iota}, ' +
            f'kappa: {self.kappa}, sigma: {self.sigma}}}, has been created')
            
    @classmethod
    def from_relation(cls, relation: np.array, es: commons.ExperimentSettings = None):
        n, m = relation.shape
        return cls(n, m, es, relation)

    def __str__(self):
        return str(self.relation)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m-1

    @property
    def relation(self):
        return self._relation[:,:self.m]

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
        return self._iota_relation[:,:self.m]


    @property
    def max_value(self):
        # max_value is used as normalizer by dividing, so it
        # should not be zero.
        maximum = np.max(self.relation)
        return 1 if maximum == 0 else maximum

    @property
    def undefined(self):
        return self.m

    @property
    def undefined_output(self):
        return np.full(self.n, np.nan)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, s):
        if s < 0:
            raise ValueError('Sigma must be a non negative number.')
        self._sigma = s
        self._sigma_scaled = abs(s*self.m)
        self._scale = normpdf(0, 0, self._sigma_scaled)

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, k):
        if k < 0:
            raise ValueError('Kappa must be a non negative number.')
        self._kappa = k

    @property
    def iota(self):
        return self._iota

    @iota.setter
    def iota(self, i):
        if i < 0:
            raise ValueError('Iota must be a non negative number.')
        self._iota = i
        self._updated = False

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, x):
        if (x < 0) or (x > self.n):
            raise ValueError('Xi must be a non negative number.')
        self._xi = x
        self._updated = False

    def register(self, cue) -> None:
        vector = self.validate(cue)
        r_io = self.to_relation(vector)
        self.abstract(r_io)

    def recognize(self, cue, validate = True):
        recognized, weights = self.recog_weights(cue, validate)
        return recognized, np.mean(weights)

    def recog_weights(self, cue, validate = True):
        vector = self.validate(cue) if validate else cue
        recognized = self._mismatches(vector) <= self.xi
        weights = self._weights(vector)
        recognized = recognized and (self.kappa*self.mean <= np.mean(weights))
        return recognized, weights

    def recall(self, cue = None):
        if cue is None:
            cue = np.full(self.n, np.nan)
        r_io, recognized, weights = self.recall_weights(cue)
        return r_io, recognized, np.mean(weights)

    def recall_weights(self, cue, validate = True):
        vector = self.validate(cue) if validate else cue
        recognized, _ = self.recog_weights(vector, validate = False)
        r_io = self.produce(vector) if recognized else np.full(self.n, self.undefined)
        weights = self._weights(r_io)
        r_io = self.revalidate(r_io)
        return r_io, recognized, weights

    def abstract(self, r_io) -> None:
        self._relation = np.where(
            self._relation == self.absolute_max_value,
            self._relation, self._relation + r_io)
        self._updated = False

    def _mismatches(self, vector):
        r_io = self.to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.n,:self.m] == 0)

    def containment(self, r_io):
        return ~r_io[:,:self.m] | self.iota_relation

    # Reduces the relation in memory to a function, given a cue
    def produce(self, cue):
        v = np.array([self.choose(i, cue[i]) for i in range(self.n)])
        return v

    # Choose a value for feature i.
    def choose(self, i, v):
        if self.is_undefined(v):
            column = self.relation[i,:]
        else:
            column = self._normalize(
                self.relation[i,:], v, self._sigma_scaled, self._scale)
        s = column.sum()
        r = s*random.random()
        for j in range(self.m):
            if r < column[j]:
                return j
            r -= column[j]
        return self.m - 1

    def _normalize(self, column, mean, std, scale):
        norm = np.array([normpdf(i, mean, std, scale) for i in range(self.m)])
        return norm*column

    def to_relation(self, cue):
        relation = np.zeros((self._n, self._m), dtype=bool)
        relation[range(self.n), cue] = True
        return relation

    def validate(self, cue):
        """ It asumes vector is an array of floats, and np.nan
            may be used to register an undefined value, but it also
            considerers any negative number or out of range number
            as undefined.
        """
        if len(cue) != self.n:
            raise ValueError('Invalid size of the input data. ' +
                f'Expected {self.n} and given {cue.size}')
        v = np.where(cue >= self.m, self.m-1, cue)
        v = np.where(v < 0, 0, v)
        v = np.nan_to_num(v, copy=True, nan=self.undefined)
        return v.astype('int')

    def revalidate(self, vector):
        v = vector.astype('float')
        return np.where(v == float(self.undefined), np.nan, v)

    def _weight(self, vector):
        return np.mean(self._weights(vector))

    def _weights(self, vector):
        weights = []
        for i in range(self.n):
            w = 0 if self.is_undefined(vector[i]) \
                else self.relation[i, vector[i]]
            weights.append(w)
        return np.array(weights)

    def is_undefined(self, value):
        return value == self.undefined

    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_entropies(self):
        totals = self.relation.sum(axis=1)  # sum of cell values by columns
        totals = np.where(totals == 0, 1, totals)
        matrix = self.relation/totals[:,None]
        matrix = -matrix*np.log2(np.where(matrix == 0.0, 1.0, matrix))
        self._entropies = matrix.sum(axis=1)

    def _update_means(self):
        sums = np.sum(self.relation, axis=1, dtype=float)
        counts = np.count_nonzero(self.relation, axis=1)
        counts = np.where(counts == 0, 1, counts)
        self._means = (sums/counts)

    def _update_iota_relation(self):
        for i in range(self._n):
            column = self._relation[i,:]
            s = np.sum(column)
            if s == 0:
                self._iota_relation[i,:] = np.zeros(self._m, dtype=int)
            else:
                count = np.count_nonzero(column)
                mean = self.iota*s/count
                self._iota_relation[i,:] = np.where(column < mean, 0, column)

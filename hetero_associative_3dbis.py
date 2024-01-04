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
        self._n = n
        self._m = m+1 # +1 to handle partial functions.
        self._p = p
        self._q = q+1 # +1 to handle partial functions.
        self._xi = es.xi
        self._absolute_max = 2**16 - 1
        self._sigma = es.sigma
        self._iota = es.iota
        self._kappa = es.kappa
        self._relation = np.zeros((self._n, self._p, self._m, self._q), dtype=int)
        self._iota_relation = np.zeros((self._n, self._p, self._m, self._q), dtype=int)
        self._entropies = np.zeros((self._n, self._p), dtype=np.double)
        self._means = np.zeros((self._n, self._p), dtype=np.double)
        self._updated = True
        # In order to accept partial functions, the borders (_m-1 and _q-1)
        # should not be zero.
        self._set_margins()
        print(f'Relational memory {self.model_name} {{n: {self.n}, p: {self.p}, ' +
            f'm: {self.m}, q: {self.q}, ' +
            f'xi: {self.xi}, iota: {self.iota}, ' +
            f'kappa: {self.kappa}, sigma: {self.sigma}}}, has been created')

    def __str__(self):
        return f'{{n: {self.n}, p: {self.p}, m: {self.m}, q: {self.q},\n{self.rel_string}}}'

    @property
    def model_name(self):
        return constants.d3_model_name
    
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
        return np.mean(self.means)

    @property
    def iota_relation(self):
        return self._full_iota_relation[:, :, :self.m, :self.q]

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
        self._sigma = sigma

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
        self._updated = False

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, x):
        if x < 0:
            raise ValueError('Xi must be a non negative number.')
        self._xi = x

    @property
    def fullness(self):
        count = np.count_nonzero(self.relation)
        total = self.n*self.m*self.p*self.q
        return count*1.0/total
    
    @property
    def rel_string(self):
        return self.relation_to_string(self.relation)

    def is_undefined(self, value, dim):
        return value == self.undefined(dim)

    def undefined(self, dim: int):
        return self.m if dim == 0 else self.q

    def undefined_function(self, dim):
        return np.full(self.cols(dim), self.undefined(dim), dtype=int)

    def alt(self, dim):
        return (dim + 1) % 2

    def cols(self, dim):
        return self.n if dim == 0 else self.p

    def rows(self, dim):
        return self.m if dim == 0 else self.q

    def register(self, cue_a, cue_b, weights_a = None, weights_b = None) -> None:
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
        recognized, weights = self.recog_full_weights(cue_a, cue_b, weights_a, weights_b, final = False)
        mean_weight = np.mean(weights)
        recognized = recognized and (self._kappa*self.mean <= mean_weight)
        return recognized, mean_weight

    def recog_full_weights(self, cue_a, cue_b, weights_a = None, weights_b = None, final = True):
        if weights_a is None:
            weights_a = np.full(len(cue_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(cue_b), fill_value=1)
        cue_a = self.validate(cue_a, 0)
        cue_b = self.validate(cue_b, 1)
        r_io = self.vectors_to_relation(cue_a, cue_b, weights_a, weights_b)
        implication = self.containment(r_io)
        recognized = np.count_nonzero(implication == 0) <= self._xi
        weights = self._weights(r_io)
        if final:
            recognized = recognized and (self._kappa*self.mean <= np.mean(weights))
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
        # If there is a column in the projection with only zeros, the cue is not recognized.
        recognized = (np.count_nonzero(np.sum(projection, axis=1) == 0) == 0)
        if not recognized:
            r_io = self.undefined_function(self.alt(dim))
            weight = 0.0
            iterations = 0
            dist_iters_mean = 0
        else:
            r_io, weights, iterations, dist_iters_mean = self.optimal_recall(cue, weights, projection, dim)
            weight = np.mean(weights)
            r_io = self.revalidate(r_io, self.alt(dim))
        return r_io, recognized, weight, projection, iterations, dist_iters_mean

    def optimal_recall(self, cue, cue_weights, projection, dim):
        r_io = None
        weights = None
        distance = float('inf')
        iterations = 0
        iter_sum = 0
        n = 0
        if not constants.d3_with_distance:
            r_io, weights = self.reduce(projection, self.alt(dim))
        else:        
            while n < constants.n_sims:
                q_io, q_ws = self.reduce(projection, self.alt(dim))
                d, iters = self.distance_recall(cue, cue_weights, q_io, q_ws, dim)
                if d < distance:
                    r_io = q_io
                    weights = q_ws
                    distance = d
                    n = 0
                else:
                    n += 1
                iterations += 1
                iter_sum += iters
        return r_io, weights, iterations, iter_sum if iterations == 0 else iter_sum/iterations


    def distance_recall(self, cue, cue_weights, q_io, q_ws, dim):
        p_io = self.project(q_io, q_ws, self.alt(dim))
        sum = self.calculate_distance(cue, cue_weights, p_io, dim)
        distance = sum
        iterations = 1
        n = 0
        while n < constants.dist_estims:
            sum += self.calculate_distance(cue, cue_weights, p_io, dim)
            iterations += 1
            d = sum/iterations
            n = 0 if abs(d-distance) > 0.01*distance else n + 1
            distance = d
        return d, iterations

    def calculate_distance(self, cue, cue_weights, p_io, dim):
        candidate, weights = self.reduce(p_io, dim)
        candidate = np.array([t[0] if self.is_undefined(t[1], dim) else t[1]
                              for t in zip(cue, candidate)])
        p = np.dot(cue_weights, weights)
        w = cue_weights*weights/p
        d = (cue-candidate)*w
        # We are not using weights in calculating distances.
        return np.linalg.norm(d)

    def abstract(self, r_io):
        self._relation = np.where(
            self._relation == self.absolute_max_value,
            self._relation, self._relation + r_io)
        self._updated = False

    def containment(self, r_io):
        return np.where((r_io == 0) | (self._full_iota_relation != 0), 1, 0)

    def project(self, cue, weights, dim):
        alt_rows = self.rows(self.alt(dim))
        integration = np.zeros(
            (self.cols(self.alt(dim)), self.rows(dim)*alt_rows), dtype=float)
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            return integration
        for i in range(cue.size):
            k = cue[i]
            if self.is_undefined(k, dim):
                continue
            w = cue.size*weights[i]/sum_weights
            projection = (self._full_iota_relation[i, :, k, :self.q] if dim == 0
                else self._full_iota_relation[:, i, :self.m, k])
            bottom = k*alt_rows
            integration[:,bottom:(bottom+alt_rows)] = projection
        return integration

    # Reduces a relation to a function
    def reduce(self, relation, dim):
        cols = self.cols(dim)
        idxs = np.array([self.choose(column, dim)
                for column in relation])
        values = []
        weights = []
        for i in range(cols):
            if idxs[i] == self.projection_undefined:
                values.append(self.undefined(dim))
                weights.append(0)
            else:
                values.append(self.projection_value(idxs[i], dim))
                weights.append(relation[i,idxs[i]])
        return np.array(values), np.array(weights)


    def choose(self, column, dim):
        """Choose a value from the column given a cue
        
        It assumes the column as a probabilistic distribution.
        """
        dist = column
        s = dist.sum()
        if s == 0:
            return self.projection_undefined
        r = s*random.random()
        for j in range(dist.size):
            if r <= dist[j]:
                return j
            r -= dist[j]
        return self.projection_undefined

    def _weights(self, r_io):
        r = r_io*np.count_nonzero(r_io)/np.sum(r_io)
        weights = np.sum(r[:, :, :self.m, :self.q] * self.relation, axis=(2,3))
        return weights

    @property    
    def projection_undefined(self):
        return self.m*self.q
    
    def projection_value(self, idx, dim):
        return idx % self.rows(dim)

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
                if total > 0:
                    matrix = relation/total
                else:
                    matrix = relation.copy()
                matrix = np.multiply(-matrix, np.log2(np.where(matrix == 0.0, 1.0, matrix)))
                self._entropies[i, j] = np.sum(matrix)
        print(f'Entropy updated to mean = {np.mean(self._entropies)}, ' 
              + f'stdev = {np.std(self._entropies)}')

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
                    self._iota_relation[i, j, :self.m, :self.q] = \
                        np.zeros((self.m, self.q), dtype=int)
                else:
                    count = np.count_nonzero(matrix)
                    threshold = self.iota*s/count
                    self._iota_relation[i, j, :self.m, :self.q] = \
                        np.where(matrix < threshold, 0, matrix)
        turned_off = np.count_nonzero(self._relation) - np.count_nonzero(self._iota_relation)
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
        threshold = self.rows(dim)
        undefined = self.undefined(dim)
        v = np.nan_to_num(cue, copy=True, nan=undefined)
        v = np.where((v < 0) | (threshold <= v), undefined, v)
        v = v.round()
        return v.astype('int')

    def revalidate(self, memory, dim):
        v = np.where(memory == self.undefined(dim), np.nan, memory)
        return v

    def vectors_to_relation(self, cue_a, cue_b, weights_a, weights_b):
        relation = np.zeros((self._n, self._p, self._m, self._q), dtype=int)
        for i in range(self.n):
            k = cue_a[i]
            for j in range(self.p):
                l = cue_b[j]
                w = math.sqrt(weights_a[i]*weights_b[j])
                relation[i, j, k, l] = int(w)
        return relation

    def _set_margins(self):
        """ Set margins to one.

        Margins are tuples (i, j, k, l) where either k = self.m or l = self.q.
        """
        self._relation[:, :, self.m, :] = np.full((self._n, self._p, self._q), 1, dtype=int)
        self._relation[:, :, :, self.q] = np.full((self._n, self._p, self._m), 1, dtype=int)
        self._iota_relation[:, :, self.m, :] = np.full((self._n, self._p, self._q), 1, dtype=int)
        self._iota_relation[:, :, :, self.q] = np.full((self._n, self._p, self._m), 1, dtype=int)

    def relation_to_string(self, a, p = ''):
        if a.ndim == 1:
            return f'{p}{a}'
        s = f'{p}[\n'
        for b in a:
            ss = self.relation_to_string(b, p + ' ')
            s = f'{s}{ss}\n'
        s = f'{s}{p}]'
        return s

    def transform(self, r):
        return r if constants.projection_transform == constants.project_same \
            else self.maximum(r) if constants.projection_transform == constants.project_maximum \
            else self.logistic(r)
    
    def maximum(self, r):
        raise NotImplementedError
                  
    def logistic(self, r):
        raise NotImplementedError
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
        
        # The value of _top depends on the hash function to be used.
        self._top = self.m * self.q
        # Additional attributes used for hashing.
        self._mbase = self.q if self.m > self.q else 1
        self._qbase = 1 if self.m > self.q else self.m

        self.xi = es.xi
        self.sigma = es.sigma
        self.iota = es.iota
        self.kappa = es.kappa
        self.relation = np.zeros((self.n, self.p, self._top+1), dtype=int)
        self._iota_relation = np.zeros((self.n, self.p, self._top+1), dtype=int)
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
    def undefined(self):
        return self._top
    
    def is_undefined(self, value):
        return value == self.undefined
    
    @property
    def entropy(self):
        """Return the entropy of the Hetero Associative Memory."""
        return np.mean(self.entropies)

    @property
    def entropies(self):
        if not self._updated:
            self._updated = self.update()
        return self._entropies

    @property
    def mean(self):
        """Return the mean of the weights in the Hetero Associative Memory"""
        return np.mean(self.means)

    @property
    def means(self):
        if not self._updated:
            self._updated = self.update()
        return self._means

    @property
    def iota_relation(self):
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation

    @property
    def fullness(self):
        count = np.count_nonzero(self.relation[:, :, :self._top])
        total = self.n*self.p*self._top
        return count/total
    
    @property
    def rel_string(self):
        return self.relation_to_string(self.relation)

    def hash(self, a, b, dim = 0):
        return a*self._mbase + b*self._qbase if dim == 0 else a*self._qbase + b*self._mbase
    
    def dehash(self, k, dim):
        return int(k/self._mbase) if dim == 0 else k % self._qbase
    
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
        recognized, weights = self.recog_full_weights(cue_a, cue_b, weights_a, weights_b, final = False)
        weight = np.mean(weights)
        recognized = recognized and (self.kappa*self.mean <= weight)
        return recognized, weight

    def recog_full_weights(self, cue_a, cue_b, weights_a, weights_b, final = True):
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
        projection_weights = np.sum(projection, axis=1)
        # If there is a column in the projection with only zeros, the cue is not recognized.
        recognized = (np.count_nonzero(projection_weights == 0) == 0)
        r_io, r_io_weights = self.reduce(projection, self.alt(dim))
        r_io_w = np.mean(r_io_weights)
        r_io = self.revalidate(r_io, self.alt(dim))
        return r_io, recognized, r_io_w, projection, 0, 0.0

    def abstract(self, r_io):
        self.relation = self.relation + r_io
        self._updated = False

    def containment(self, r_io):
        c = np.where((r_io[:, :, :self._top] == 0) | (self.iota_relation[:, :, :self._top] != 0),1,0)
        return(c)

    def project(self, cue, weights, dim):
        projection = np.zeros((self.cols(self.alt(dim)), self._top), dtype=int)
        chosen = self.filter_relation(cue, weights, dim)
        for j in range(self.cols(self.alt(dim))):
            for k in range(self._top):
                w = np.sum(chosen[:, j, k] if dim == 0 else chosen[j, :, k])
                projection[j, k] = w
        return projection

    def filter_relation(self, cue, weights, dim):
        chosen = np.zeros(self.relation.shape, dtype=float)
        for i in range(self.cols(dim)):
            value = cue[i]
            weight = weights[i]
            if self.is_undefined(value):
                continue
            for j in range(self.cols(self.alt(dim))):
                a = i if dim == 0 else j
                b = j if dim == 0 else i
                for k in range(self.rows(self.alt(dim))):
                    idx = self.hash(value, k, dim)
                chosen[a, b, idx] = self.relation[a, b, idx] * weight
        return chosen

    # Reduces a relation to a function
    def reduce(self, projection, dim):
        cols = self.cols(dim)
        vw = [self.choose(column, dim) for column in projection]
        v = np.array([t[0] for t in vw])
        w = np.array([t[1] for t in vw])
        return v, w

    
    def choose(self, distribution, dim):
        """Choose a value from a probability distribution"""
        s = distribution.sum()
        if s == 0:
            return self.dehash(random.randrange(distribution.size), dim), 0
        r = s*random.random()
        for j in range(self._top):
            if r <= distribution[j]:
                return self.dehash(j, dim), distribution[j]
            r -= distribution[j]
        return self.undefined, 0

    def _weights(self, r_io):
        r = r_io*np.count_nonzero(r_io)/np.sum(r_io)
        weights = np.sum(r * self.relation, axis=2)
        return weights
        
    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_entropies(self):
        for i in range(self.n):
            for j in range(self.p):
                r = self.relation[i, j, :self._top]
                total = np.sum(r)
                if total > 0:
                    matrix = r/total
                else:
                    matrix = r.copy()
                matrix = np.multiply(-matrix, np.log2(np.where(matrix == 0.0, 1.0, matrix)))
                self._entropies[i, j] = np.sum(matrix)
        print(f'Entropy updated to mean = {np.mean(self._entropies)}, ' 
              + f'stdev = {np.std(self._entropies)}')

    def _update_means(self):
        for i in range(self.n):
            for j in range(self.p):
                r = self.relation[i, j, :self._top]
                count = np.count_nonzero(r)
                count = 1 if count == 0 else count
                self._means[i,j] = np.sum(r)/count

    def _update_iota_relation(self):
        for i in range(self.n):
            for j in range(self.p):
                column = self.relation[i, j, :].copy()
                s = np.sum(column)
                if s > 0:
                    count = np.count_nonzero(column)
                    threshold = self.iota*s/count
                    column= np.where(column < threshold, 0, column)
                self._iota_relation[i, j, :] = column
        turned_off = np.count_nonzero(self.relation) - np.count_nonzero(self._iota_relation)
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
        undefined = self.undefined
        v = np.nan_to_num(cue, copy=True, nan=undefined)
        v = np.where((v < 0) | (threshold <= v), undefined, v)
        v = v.round()
        return v.astype('int')
    
    def revalidate(self, memory, dim):
        v = np.where(memory == self.undefined, np.nan, memory)
        return v

    def vectors_to_relation(self, cue_a, cue_b, weights_a, weights_b):
        relation = np.zeros((self.n, self.p, self._top+1), dtype=int)
        for i in range(self.n):
            a = cue_a[i]
            for j in range(self.p):
                b = cue_b[j]
                if self.is_undefined(a) or self.is_undefined(b):
                    continue
                k = self.hash(a,b)
                w = weights_a[i]*weights_b[j]
                relation[i, j, k] = w
        return relation

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
        raise NameError('Method "maximum" is undefined')            

    def logistic(self, r):
        raise NameError('Method "logistic" is undefined')            

    def get_random_string(self):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        random_string = ''.join(random.choice(letters) for i in range(constants.random_string_length))
        return random_string

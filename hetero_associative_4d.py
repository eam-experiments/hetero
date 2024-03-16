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
import numpy as np
import tensorflow as tf
import commons

class HeteroAssociativeMemory4D:
    def __init__(self, n: int, p: int, m: int, q: int,
        es: commons.ExperimentSettings, fold, nm_qd = None, pq_qd = None,
        prototypes = None):
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

        prototypes: A list of arrays of prototpyes for the domains
            defined by (n,m), and (p,q), or a list of None.
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
        self._prototypes = self.validate_prototypes(prototypes)
        self._entropies = np.zeros((self._n, self._p), dtype=np.double)
        self._means = np.zeros((self._n, self._p), dtype=np.double)
        self._updated = True
        # In order to accept partial functions, the borders (_m-1 and _q-1)
        # should not be zero.
        self._set_margins()

        # Set quantizers/dequantizers per dimension.
        self.qudeqs = [nm_qd, pq_qd]

        # Retrieve the classifiers.
        self.classifiers = []
        for dataset in commons.datasets:
            model_prefix = commons.model_name(dataset, es)
            filename = commons.classifier_filename(model_prefix, es, fold)
            classifier = tf.keras.models.load_model(filename)
            self.classifiers.append(classifier)

        print(f'Relational memory {self.model_name} {{n: {self.n}, p: {self.p}, ' +
            f'm: {self.m}, q: {self.q}, ' +
            f'xi: {self.xi}, iota: {self.iota}, ' +
            f'kappa: {self.kappa}, sigma: {self.sigma}}}, has been created')

    def __str__(self):
        return f'{{n: {self.n}, p: {self.p}, m: {self.m}, q: {self.q},\n{self.rel_string}}}'

    @property
    def model_name(self):
        return commons.d4_model_name
    
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

    def is_partial(self, f, dim):
        u = self.undefined(dim)
        for v in f:
            if v == u:
                return True
        return False
    
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

    def recall_from_left(self, cue, method = commons.recall_with_search,
            euc = None, weights = None, label = None):
        if weights is None:
            weights = np.full(len(cue), fill_value=1)
        return self._recall(cue, weights, label, 0)

    def recall_from_right(self, cue, method = commons.recall_with_search,
            euc = None, weights = None, label = None):
        if weights is None:
            weights = np.full(len(cue), fill_value=1)
        return self._recall(cue, weights, label, 1)

    def _recall(self, cue, weights, label, dim):
        cue = self.validate(cue, dim)
        projection = self.project(cue, weights, dim)
        recognized = (np.count_nonzero(np.sum(projection, axis=1) == 0) == 0)
        if recognized:
            projection = self.transform(projection, label, dim)
            # If there is a column in the projection with only zeros, the cue is not recognized.
            recognized = (np.count_nonzero(np.sum(projection, axis=1) == 0) == 0)
        if not recognized:
            r_io = self.undefined_function(self.alt(dim))
            weight = 0.0
            stats = [0, 0, 0.0, 0.0]
        else:
            r_io, weights, stats = \
                    self.optimal_recall(cue, weights, label, projection, dim)
            if r_io is None:
                recognized = False
                r_io = self.undefined_function(self.alt(dim))
                weight = 0.0
            else:
                weight = np.mean(weights)
                r_io = self.revalidate(r_io, self.alt(dim))
        return r_io, recognized, weight, projection, stats

    def optimal_recall(self, cue, cue_weights, label, projection, dim):
        sampling_iterations = 0
        p = 1.0
        step = p / commons.sample_size if commons.sample_size > 0 else p
        last_update = 0
        r_io, weights = self.get_initial_cue(cue, cue_weights, label, projection, dim)
        distance, _ = self.distance_recall(cue, cue_weights, label, r_io, weights, dim)
        visited = [r_io]
        for k, beta in zip(range(commons.sample_size), np.linspace(1.0, self.sigma, commons.sample_size)):
            # s = self.rows(self.alt(dim)) * beta
            excluded = None # self.random_exclusion(r_io, p)
            s_projection = projection # self.adjust(projection, r_io, s)
            q_io, q_ws = self.reduce(s_projection, self.alt(dim), excluded)
            j = 0
            while self.already_visited(q_io, visited) and (j < commons.dist_estims):
                q_io, q_ws = self.reduce(s_projection, self.alt(dim), excluded)
                j += 1
            if j == commons.dist_estims:
                continue
            visited.append(q_io)
            d, _ = self.distance_recall(cue, cue_weights, label, q_io, q_ws, dim)
            if d < distance:
                r_io = q_io
                weights = q_ws
                distance = d
                sampling_iterations += 1
                last_update = k
            p -= step
        distance2 = distance
        better_found = True
        k = commons.sample_size
        search_iterations = 0
        sampling_io = r_io
        sampling_ws = weights
        while better_found:
            neighbors = self.neighborhood(projection, r_io, self.alt(dim))
            better_found = False
            p_io = None
            p_ws = None
            while neighbors:
                t = random.choice(neighbors)
                i = t[0]
                v = t[1]
                neighbors.remove(t)
                q_io = np.array([r_io[j] if j != i else v for j in range(self.cols(self.alt(dim)))])
                q_ws = self.weights_in_projection(projection, q_io, self.alt(dim))
                d, _ = self.distance_recall(cue, cue_weights, label, q_io, q_ws, dim)
                k += 1
                if d < distance2:
                    p_io = q_io
                    p_ws = q_ws
                    distance2 = d
                    search_iterations += 1
                    last_update = k
                    better_found = True
                    break
            if better_found:
                r_io = p_io
                weights = p_ws
        diffs, length = self.functions_distance(sampling_io, sampling_ws, r_io, weights)
        return r_io, weights, [sampling_iterations, search_iterations,
                last_update, distance2, (distance2- distance), diffs, length]

    def get_initial_cue(self, cue, cue_weights, label, projection, dim):
        return self.reduce(projection, self.alt(dim))
        if self._prototypes[self.alt(dim)] is None:
            r_io, r_ws = self.reduce(projection, self.alt(dim))
            return r_io, r_ws
        distance = float('inf')
        candidate = None
        candidate_weights = None
        for proto in self._prototypes[self.alt(dim)]:
            ws = []
            for i in range(proto.size):
                if self.is_undefined(proto[i], self.alt(dim)) \
                        or (projection[i, proto[i]] == 0):
                    ws = []
                    break
                else:
                    ws.append(projection[i, proto[i]])
            if not ws:
                continue
            else:
                ws = np.array(ws)
                d, _ = self.distance_recall(cue, cue_weights, proto, ws, dim)
                if d < distance:
                    candidate = proto
                    candidate_weights = ws
                    distance = d
        if candidate is None:
            candidate, candidate_weights = self.reduce(projection, self.alt(dim))
        return candidate, candidate_weights

    def distance_recall(self, cue, cue_weights, label, q_io, q_ws, dim):
        p_io = self.project(q_io, q_ws, self.alt(dim))
        distance = self.calculate_distance(cue, cue_weights, p_io, dim)
        return distance, 0

    def calculate_distance(self, cue, cue_weights, p_io, dim):
        distance = 0.0
        for v, w, column in zip(cue, cue_weights, p_io):
            s = np.sum(column)
            ps = column if s == 0.0 else column/np.sum(column)
            d = np.dot(np.square(np.arange(self.rows(dim))-v),ps)*w
            distance += d
        return distance / np.sum(cue_weights)
    
    def functions_distance(self, p_io, p_ws, q_io, q_ws):
        abs = np.abs(p_io - q_io)
        diff = np.sum(abs)
        length = np.max(abs)
        return diff, length

    def presence_entropy(self, cue, cue_weights, label, q_io, q_ws, dim):
        p_io = self.project(q_io, q_ws, self.alt(dim))
        presence = self.label_presence(p_io, label, dim)
        entropy = self.projection_entropy(p_io, dim)
        return presence, entropy
    
    def abstract(self, r_io):
        self._relation = np.where(
            self._relation == self.absolute_max_value,
            self._relation, self._relation + r_io)
        self._updated = False

    def containment(self, r_io):
        return np.where((r_io == 0) | (self._full_iota_relation != 0), 1, 0)

    def project(self, cue, weights, dim):
        integration = np.zeros((self.cols(self.alt(dim)), self.rows(self.alt(dim))), dtype=float)
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            return integration
        first = True
        w = cue.size*weights/sum_weights
        for i in range(cue.size):
            k = cue[i]
            if self.is_undefined(k, dim):
                continue
            projection = (self._full_iota_relation[i, :, k, :self.q] if dim == 0
                else self._full_iota_relation[:, i, :self.m, k])
            if first:
                integration = w[i]*projection
                first = False
            else:
                integration = np.where((integration == 0) | (projection == 0),
                        0, integration + w[i]*projection)
        return integration

    # Reduces a relation to a function
    def reduce(self, relation, dim, excluded = None):
        cols = self.cols(dim)
        v = np.array([self.choose(column, dim) for column in relation]) \
            if excluded is None else \
                np.array([self.choose(column, dim, exc) for column, exc in zip(relation, excluded)])
        weights = []
        for i in range(cols):
            if self.is_undefined(v[i], dim):
                weights.append(0)
            else:
                weights.append(relation[i, v[i]])
        return v, np.array(weights)

    def choose(self, column, dim, excluded = None):
        """Choose a value from the column given a cue
        
        It assumes the column as a probabilistic distribution.
        """
        s = column.sum()
        if s == 0:
            return self.undefined(dim)
        if (excluded is not None):
            if s > column[excluded]:
                s -= column[excluded]
            else:
                excluded = None
        r = s*random.random()
        for j in range(column.size):
            if (excluded is not None) and (j == excluded):
                continue
            if r <= column[j]:
                return j
            r -= column[j]
        return self.undefined(dim)

    def neighborhood(self, projection, r_io, dim):
        neigh = []
        rows = self.rows(dim)-1
        for i in range(self.cols(dim)):
            column = projection[i]
            value = r_io[i]
            if (value < rows) and (column[value+1] > 0):
                neigh.append((i, value+1))
            if (0 < value) and (column[value-1] > 0):
                neigh.append((i, value-1))
        return neigh

    def adjust(self, projection, cue, s):
        if cue is None:
            return projection
        s_projection = []
        for column, mean in zip(projection, cue):
            adjusted = self.ponderate(column, mean, s)
            s_projection.append(adjusted)
        return np.array(s_projection)

    def ponderate(self, column, mean, s):
        norm = np.array([self.normpdf(i, mean, s)/self.normpdf(0, 0, s) for i in range(column.size)])
        return norm*column
    
    def _weights(self, r_io):
        r = r_io*np.count_nonzero(r_io)/np.sum(r_io)
        weights = np.sum(r[:, :, :self.m, :self.q] * self.relation, axis=(2,3))
        return weights
        
    def weights_in_projection(self, projection, q_io, dim):
        return projection[np.arange(self.cols(dim)), q_io]
    
    def random_exclusion(self, cue, p):
        excluded = []
        for v in cue:
            r = random.random()
            excluded.append(v if r < p else None)
        return excluded

    def complement(self, relation):
        maximum = np.max(relation)
        return maximum - relation
        
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
        if (len(cue.shape) < 1) or (len(cue.shape) > 2):
            raise ValueError(f'Unexpected shape of cue(s): {cue.shape}.')
        if len(cue.shape) == 1:
            if cue.size != expected_length:
                raise ValueError('Invalid lenght of the input data. Expected ' +
                        f'{expected_length} and given {cue.size}')
        elif cue.shape[1] != expected_length:
            raise ValueError(f'Expected shape (n, {expected_length}) ' +
                    f'but got shape {cue.shape}')
        threshold = self.rows(dim)
        undefined = self.undefined(dim)
        v = np.nan_to_num(cue, copy=True, nan=undefined)
        v = np.where((v < 0) | (threshold <= v), undefined, v)
        v = v.round()
        return v.astype('int')

    def revalidate(self, memory, dim):
        v = np.where(memory == self.undefined(dim), np.nan, memory)
        return v

    def validate_prototypes(self, prototypes):
        if prototypes is None:
            return [None, None]
        protos = []
        for dim in range(2):
            protos.append(None if prototypes[dim] is None \
                    else self.validate(prototypes[dim], dim))
        return protos

    def vectors_to_relation(self, cue_a, cue_b, weights_a, weights_b):
        relation = np.zeros((self._n, self._p, self._m, self._q), dtype=int)
        for i in range(self.n):
            k = cue_a[i]
            for j in range(self.p):
                label = cue_b[j]
                w = math.sqrt(weights_a[i]*weights_b[j])
                relation[i, j, k, label] = int(w)
        return relation

    def _set_margins(self):
        """ Set margins to one.

        Margins are tuples (i, j, k, l) where either k = self.m or l = self.q.
        """
        self._relation[:, :, self.m, :] = np.full((self._n, self._p, self._q), 1, dtype=int)
        self._relation[:, :, :, self.q] = np.full((self._n, self._p, self._m), 1, dtype=int)
        self._iota_relation[:, :, self.m, :] = np.full((self._n, self._p, self._q), 1, dtype=int)
        self._iota_relation[:, :, :, self.q] = np.full((self._n, self._p, self._m), 1, dtype=int)

    def labels_in_projection(self, projection, label, dim):
        counts = np.zeros(commons.n_labels, dtype=int)
        classifier = self.classifiers[self.alt(dim)]
        for lbl in commons.all_labels:
            proto = self._prototypes[self.alt(dim)][lbl]
            s = self.rows(self.alt(dim)) * self.sigma
            p = self.adjust(projection, proto, s)
            if (np.count_nonzero(np.sum(projection, axis=1) == 0) != 0):
                continue
            candidates = []
            for i in range(commons.presence_iterations):
                r_io, _ = self.reduce(p, self.alt(dim))
                if not self.is_partial(r_io, self.alt(dim)):
                    candidates.append(r_io)
            if len(candidates) > 0:
                candidates = self.rsize_recalls(np.array(candidates), self.alt(dim))
                classification = np.argmax(classifier(candidates, training=False), axis=1)
                for c in classification:
                    counts[lbl] += (c == lbl)
        sorted_labels = np.argsort(counts)[::-1]
        best_other = 0
        for lbl in sorted_labels:
            if lbl != label:
                best_other = lbl
                break
        stats = {label: counts[label]/commons.presence_iterations,
                best_other: counts[best_other]/commons.presence_iterations}
        return stats
    
    def label_presence(self, projection, label, dim):
        r_ios = np.zeros((commons.dist_estims, self.cols(dim)), dtype = int)
        for i in range(commons.dist_estims):
            r_io, _ = self.reduce(projection, dim)
            r_ios[i] = np.array(r_io, dtype=int)
        r_ios = self.rsize_recalls(r_ios, dim)
        classifier = self.classifiers[dim]
        classification = np.argmax(classifier(r_ios, training=False), axis=1)
        labels, counts = np.unique(classification, return_counts=True)
        frequencies = dict(zip(labels, counts))
        maximum = max(frequencies.values())
        presence = frequencies[label] if label in frequencies.keys() else 0
        return presence/commons.dist_estims if presence == maximum else 0.0

    def projection_entropy(self, projection, dim):
        entropies = []
        for i in range(self.cols(dim)):
            total = np.sum(projection[i])
            if total > 0:
                column = projection[i]/total
            else:
                column = projection[i].copy()
            column = np.multiply(-column, np.log2(np.where(column == 0.0, 1.0, column)))
            entropies.append(np.sum(column))
        return np.mean(entropies)

    def rsize_recalls(self, recalls, dim):
        return self.qudeqs[dim].dequantize(recalls, self.rows(dim))

    def relation_to_string(self, a, p = ''):
        if a.ndim == 1:
            return f'{p}{a}'
        s = f'{p}[\n'
        for b in a:
            ss = self.relation_to_string(b, p + ' ')
            s = f'{s}{ss}\n'
        s = f'{s}{p}]'
        return s

    def already_visited(self, r_io, visited):
        for q_io in visited:
            if np.array_equal(r_io, q_io):
                return True
        return False
    
    def transform(self, r, label, dim):
        match commons.projection_transform:
            case commons.project_same:
                return r
            case commons.project_maximum:
                return self.maximum(r)
            case commons.project_logistic:
                return self.logistic(r)
            case commons.project_prototype:
                return self.adjust_by_proto(r, label, dim)
            case _:
                raise ValueError('Unexpected value of commons.projection_transform: ' +
                                 f'{commons.projection_transform}.')
    
    def adjust_by_proto(self, r, label, dim):
        stats = self.labels_in_projection(r, label, dim)
        other = [k for k in stats][1]
        lbl = label if stats[label] >= stats[other] else other
        proto = self._prototypes[self.alt(dim)][lbl]
        s = self.rows(self.alt(dim)) * self.sigma
        q = self.adjust(r, proto, s)
        return q

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
    
    def normpdf(self, x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom


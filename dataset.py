# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

import gzip
import numpy as np
import os
import random
import commons

# This code is an abstraction for the MNIST and MNIST Fashion datasets.
columns = 28
rows = 28

_TRAINING_SEGMENT = 0
_FILLING_SEGMENT = 1
_TESTING_SEGMENT = 2

def get_training(dataset, fold):
    return _get_segment(dataset, _TRAINING_SEGMENT, fold)

def get_filling(dataset, fold):
    return _get_segment(dataset, _FILLING_SEGMENT, fold)

def get_testing(dataset, fold, noised = False):
    return _get_segment(dataset, _TESTING_SEGMENT, fold, noised)

def _get_segment(dataset, segment, fold, noised = False):
    if (_get_segment.data is None):
        _get_segment.data = _load_dataset(dataset, commons.data_path)
    print('Delimiting segment of data.')
    # We assume the dataset is balanced
    data, labels = _get_data_in_range(segment, _get_segment.data, fold, noised)
    return data, labels

_get_segment.data = None

def noised(data, percent):
    print(f'Adding {percent}% noise to data.')
    copy = np.zeros(data.shape, dtype=float)
    n = 0
    for i in range(len(copy)):
        copy[i] = _noised(data[i], percent)
        n += 1
        commons.print_counter(n, 10000, step=100)
    return copy

def _noised(image, percent):
    copy = np.array([row[:] for row in image])
    total = round(columns*rows*percent/100.0)
    noised = []
    while len(noised) < total:
        i = random.randrange(rows)
        j = random.randrange(columns)
        if (i, j) in noised:
            continue
        value = random.randrange(255)
        copy[i,j] = value
        noised.append((i,j))
    return copy       

def _load_dataset(dataset, path):
    dirname = os.path.join(path, dataset)
    data, noised_data, labels = _preprocessed_dataset(dirname)
    if (data is None) or (noised_data is None) or (labels is None):
        data_train, labels_train = _load_mnist_like(dirname, kind='train')
        data_test, labels_test = _load_mnist_like(dirname, kind='t10k')
        data = np.concatenate((data_train, data_test), axis=0).astype(dtype=float)
        noised_data = noised(data, commons.noise_percent)
        labels = np.concatenate((labels_train, labels_test), axis=0)
        data, noised_data, labels = _shuffle(data, noised_data, labels)
        _save_dataset(dirname, data, noised_data, labels)
    data = _split_by_labels(data, noised_data, labels)
    return data

def _preprocessed_dataset(dirname):
    data_fname = os.path.join(dirname, commons.prep_data_fname)
    noised_fname = os.path.join(dirname, commons.pred_noised_data_fname)
    labels_fname = os.path.join(dirname, commons.prep_labels_fname)
    data = None
    noised = None
    labels = None
    try:
        data = np.load(data_fname)
        noised = np.load(noised_fname)
        labels = np.load(labels_fname).astype('int')
        print('Preprocessed dataset exists, so it is used.')
    except:
        print('Preprocessed dataset does not exist.')
    return data, noised, labels

def _save_dataset(dirname, data, noised, labels):
    print('Saving preprocessed dataset')
    data_fname = os.path.join(dirname, commons.prep_data_fname)
    noised_fname = os.path.join(dirname, commons.pred_noised_data_fname)
    labels_fname = os.path.join(dirname, commons.prep_labels_fname)
    np.save(data_fname, data)
    np.save(noised_fname, noised)
    np.save(labels_fname, labels)

def _load_mnist_like(dirname, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(dirname, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(dirname, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),
            dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),
            dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
    return images, labels

def _shuffle(data, noised, labels):
    print('Shuffling data and labels')
    tuples = [(data[i], noised[i], labels[i]) for i in range(len(labels))]
    random.shuffle(tuples)
    data = np.array([p[0] for p in tuples])
    noised = np.array([p[1] for p in tuples])
    labels = np.array([p[2] for p in tuples], dtype=int)
    return data, noised, labels

def _split_by_labels(data, noised, labels):
    data_per_label = {}
    for l, d, n in zip(labels, data, noised):
        if l in data_per_label.keys():
            data_per_label[l].append((l, d, n))
        else:
            data_per_label[l] = [(l, d, n)]
    return data_per_label

def _get_data_in_range(segment, data_per_label, fold, noised):
    data = []
    for label in commons.all_labels:
        total = len(data_per_label[label])
        training = total*commons.nn_training_percent
        filling = total*commons.am_filling_percent
        testing = total*commons.am_testing_percent
        step = total / commons.n_folds
        i = fold * step
        j = i + training
        k = j + filling
        l = k + testing
        i = int(i)
        j = int(j) % total
        k = int(k) % total
        l = int(l) % total
        n, m = None, None
        if segment == _TRAINING_SEGMENT:
            n, m = i, j
        elif segment == _FILLING_SEGMENT:
            n, m = j, k
        elif segment == _TESTING_SEGMENT:
            n, m = k, l
        dpl = commons.get_data_in_range(data_per_label[label], n, m)
        data += dpl
    random.shuffle(data)
    labels = np.array([d[0] for d in data])
    i = 2 if noised else 1
    data = np.array([d[i] for d in data])
    return data, labels


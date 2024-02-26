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

"""Entropic Hetero-Associative Memory Experiments

Usage:
  eam -h | --help
  eam --dims=N (-n <dataset> | -f <dataset> | -d <dataset> | -s <dataset> | -c <dataset> | -e | -r | -v | -w)
    [--relsmean=MEAN] [--relsstdv=STDV] [--runpath=PATH] [ -p ] [ -l (en | es) ]

Options:
  -h    Show this screen.
  -n    Trains the neural network for MNIST (mnist) or Fashion (fashion).
  -f    Generates Features for MNIST (mnist) or Fashion (fashion).
  -c    Characterizes features and generate prototypes per class.
  -d    Calculate distances intra/inter classes of features.
  -s    Run separated tests of memories performance for MNIST y Fashion.
  -e    Evaluation of recognition of hetero-associations.
  -r    Evaluation of hetero-recalling.
  -v    Validation of hetero-recalling using filling data (easier).
  -w    Validation of hetero-recalling using testing data (harder).
  --dims=N         Dimensions of the model (either 3 or 4)
  --relsmean=MEAN   Average number of relations per data element.
  --relsstdv=STDV   Standard deviation of the number of relations per data element.
  --runpath=PATH    Path to directory where everything will be saved [default: runs]
  -p    Load prototypes for MNIST and Fashion.
  -l    Chooses Language for graphs.
"""

import os
import sys
import gc
import gettext
import json
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import tensorflow as tf
import png
from docopt import docopt, DocoptExit
# This disables importation of this file but allows selection of model
args = docopt(__doc__)
import commons
import dataset as ds
import neural_net
from custom_set import CustomSet
from associative import AssociativeMemory

dims = int(args['--dims'])
if dims == 3:
    from hetero_associative_3d import HeteroAssociativeMemory3D as HeteroAssociativeMemory
elif dims == 4:
    from hetero_associative_4d import HeteroAssociativeMemory4D as HeteroAssociativeMemory
else:
    raise DocoptExit('Invalid number of dimensions (only 3 or 4 are accepted)')

sys.setrecursionlimit(10000)

# This allows the production of graphs in multiple languages
# (English is the default).
gettext.bindtextdomain('eam', 'locale')
gettext.textdomain('eam')
_ = gettext.gettext

# Categories in binary confussion matrix
TP = (0, 0)
FN = (0, 1)
FP = (1, 0)
TN = (1, 1)


def plot_prerec_graph(pre_mean, rec_mean, ent_mean, pre_std, rec_std, dataset,
                   es, acc_mean=None, acc_std=None,
                   prefix='', xlabels=None,
                   xtitle=None, ytitle=None):
    """ Plots a precision and recall graph.
    
        It can also include accuracy, and it can be used to to graph any
        measure in percentages.
    """
    plt.figure(figsize=(6.4, 4.8))
    full_length = 100.0
    step = 0.1
    if xlabels is None:
        xlabels = commons.memory_sizes
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step

    # Gives space to fully show markers in the top.
    ymax = full_length + 2

    # Replace undefined precision with 1.0.
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)

    plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(x, rec_mean, fmt='b--s', yerr=rec_std, label=_('Recall'))
    if (acc_mean is not None) and (acc_std is not None):
        plt.errorbar(x, acc_mean, fmt='g--d',
                     yerr=acc_std, label=_('Accuracy'))
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)
    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None:
        ytitle = _('Percentage')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc='best')
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'mycolors', ['cyan', 'purple'])
    z = [[0, 0], [0, 0]]
    levels = np.arange(0.0, xmax, step)
    cs3 = plt.contourf(z, levels, cmap=cmap)

    cbar = plt.colorbar(cs3, orientation='horizontal')
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_('Entropy'))

    fname = prefix + 'graph_metrics-' + dataset + _('-english')
    graph_filename = commons.picture_filename(fname, es)
    plt.savefig(graph_filename, dpi=600)
    plt.close()


def plot_behs_graph(no_response, no_correct, correct, dataset, es, xtags=None, prefix=''):
    """ Plots the behaviours graph.
    
        A behaviour graph is a stacked bars graph that representes three exclusive types
        of all possible responses: no response, no correct response, and correct response.
    """
    plt.clf()
    print('Behaviours: ')
    print(f'No response: {no_response}')
    print(f'No correct response: {no_correct}')
    print(f'Correct response: {correct}')
    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + correct[i])/100.0
        no_response[i] /= total
        no_correct[i] /= total
        correct[i] /= total
    full_length = 100.0
    step = 0.1
    if xtags is None:
        xtags = commons.memory_sizes
    main_step = full_length/len(xtags)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5       # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label=_('Correct response'))
    cumm = np.array(correct)
    plt.bar(x, no_correct, width, bottom=cumm, label=_('No correct response'))
    cumm += np.array(no_correct)
    plt.bar(x, no_response, width, bottom=cumm, label=_('No response'))

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, xtags)

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Labels'))

    plt.legend(loc='best')
    plt.grid(axis='y')

    fname = prefix + 'graph_behaviours-' + dataset + _('-english')
    graph_filename = commons.picture_filename(fname, es)
    plt.savefig(graph_filename, dpi=600)
    plt.close()


def plot_features_graph(domain, means, stdevs, dataset, es):
    """ Draws the characterist shape of features per label.

    The graph is a dots and lines graph with error bars denoting standard deviations.
    """
    ymin = np.PINF
    ymax = np.NINF
    for i in commons.all_labels:
        yn = (means[i] - stdevs[i]).min()
        yx = (means[i] + stdevs[i]).max()
        ymin = ymin if ymin < yn else yn
        ymax = ymax if ymax > yx else yx
    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = commons.label_formats
    for i in commons.all_labels:
        plt.clf()
        plt.figure(figsize=(12,5))
        plt.errorbar(xrange, means[i], fmt=fmts[i], yerr=stdevs[i], label=str(i))
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels='')
        plt.xlabel(_('Features'))
        plt.ylabel(_('Values'))
        plt.legend(loc='right')
        plt.grid(axis='y')
        filename = commons.features_name(dataset, es) + '-' + str(i).zfill(3) + _('-english')
        plt.savefig(commons.picture_filename(filename, es), dpi=600)
    plt.close()


def plot_confusion_matrix(matrix, tags, dataset, es, prefix='', vmin=0.0, vmax=None):
    """ Plots the confusion matrix of labels vs predictions."""
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    if vmax is None:
        vmax = np.max(matrix)
    seaborn.heatmap(matrix, xticklabels=tags, yticklabels=tags,
                    vmin=vmin, vmax=vmax, annot=False, cmap='Blues')
    plt.xlabel(_('Prediction'))
    plt.ylabel(_('Label'))
    fname = prefix + commons.matrix_suffix + '-' + dataset + _('-english')
    filename = commons.picture_filename(fname, es)
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_relation(relation, prefix, xlabel = 'Characteristics', 
        ylabel = 'Values', es=None, fold=None):
    """ Plots a relation (table) as a heat map."""
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(np.transpose(relation), annot=False, cmap='coolwarm')
    plt.xlabel(_(xlabel))
    plt.ylabel(_(ylabel))
    if es is None:
        es = commons.ExperimentSettings()
    filename = commons.picture_filename(prefix, es, fold)
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_distances(distances, prefix, es=None, fold=None):
    """ Plots a matrix of distances between categories (labels) in a dataset."""
    plot_relation(distances, prefix, xlabel='Label', ylabel='Label', es=es, fold=fold)


def get_min_max(a : np.ndarray, percentiles = False):
    """Produces a desirable minimum and maximum values for features."""
    maximum = np.max(a)
    minimum = np.min(a)
    min_percentile = np.percentile(a, commons.minimum_percentile)
    max_percentile = np.percentile(a, commons.maximum_percentile)
    median = np.median(a)
    mean = np.mean(a)
    stdv = np.std(a)
    print(f'Min_maxs array stats: min = {minimum}, ' +
          f'{commons.minimum_percentile}% = {min_percentile}, ' +
            f'median = {median}, {commons.maximum_percentile}% = {max_percentile}, ' +
            f'max = {maximum}; mean = {mean}, stdev = {stdv}')
    return (min_percentile, max_percentile) if percentiles else (minimum, maximum)

def features_distance(f, g):
    """ Calculates euclidean distance between two arrays of features."""
    return np.linalg.norm(f - g)


def stats_measures(filling_features, filling_labels,
                   testing_features, testing_labels):
    """ Calculates mean and standard deviation for each feature element per label."""
    filling_fpl = {}
    testing_fpl = {}
    for label in range(commons.n_labels):
        filling_fpl[label] = []
        testing_fpl[label] = []
    for f, l in zip(filling_features, filling_labels):
        filling_fpl[l].append(f)
    for f, l in zip(testing_features, testing_labels):
        testing_fpl[l].append(f)
    means = np.zeros((commons.n_labels+1, 2), dtype=float)
    stdvs = np.zeros((commons.n_labels+1, 2), dtype=float)
    for l in range(commons.n_labels):
        means[l, 0] = np.mean(filling_fpl[l])
        means[l, 1] = np.mean(testing_fpl[l])
        stdvs[l, 0] = np.std(filling_fpl[l])
        stdvs[l, 1] = np.std(testing_fpl[l])
    means[commons.n_labels, 0] = np.mean(filling_features)
    means[commons.n_labels, 1] = np.mean(testing_features)
    stdvs[commons.n_labels, 0] = np.std(filling_features)
    stdvs[commons.n_labels, 1] = np.std(testing_features)
    return means, stdvs


def distance_matrices(filling_features, filling_labels,
                      testing_features, testing_labels):
    ff_dist = {}
    ft_dist = {}
    for l1 in range(commons.n_labels):
        for l2 in range(commons.n_labels):
            ff_dist[(l1, l2)] = []
            ft_dist[(l1, l2)] = []
    f_len = len(filling_labels)
    t_len = len(testing_labels)
    counter = 0
    for i in range(f_len):
        for j in range(f_len):
            if i != j:
                l1 = filling_labels[i]
                l2 = filling_labels[j]
                d = features_distance(filling_features[i], filling_features[j])
                ff_dist[(l1, l2)].append(d)
        for j in range(t_len):
            l1 = filling_labels[i]
            l2 = testing_labels[j]
            d = features_distance(filling_features[i], testing_features[j])
            ft_dist[(l1, l2)].append(d)
        commons.print_counter(counter, 1000, 100)
        counter += 1
    print(' end.')
    ff_means = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    ff_stdvs = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    ft_means = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    ft_stdvs = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    for l1 in range(commons.n_labels):
        for l2 in range(commons.n_labels):
            mean = np.mean(ff_dist[(l1, l2)])
            stdv = np.std(ff_dist[(l1, l2)])

            ff_means[l1, l2] = mean
            ff_stdvs[l1, l2] = stdv
            mean = np.mean(ft_dist[(l1, l2)])
            stdv = np.std(ft_dist[(l1, l2)])
            ft_means[l1, l2] = mean
            ft_stdvs[l1, l2] = stdv
    means = np.concatenate((ff_means, ft_means), axis=1)
    stdvs = np.concatenate((ff_stdvs, ft_stdvs), axis=1)
    return means, stdvs


def msize_features(features, msize, min_value, max_value):
    return np.round((msize-1)*(features-min_value) / (max_value-min_value)).astype(int)


def rsize_recall(recall, msize, min_value, max_value):
    if msize == 1:
        return np.full(recall.size, (min_value + max_value)/2.0)
    return (max_value - min_value) * recall.astype(dtype=float) / (msize - 1.0) + min_value


def features_per_fold(dataset, es, fold):
    suffix = commons.filling_suffix
    filling_features_filename = commons.features_name(dataset, es) + suffix
    filling_features_filename = commons.data_filename(
        filling_features_filename, es, fold)
    filling_labels_filename = commons.labels_name(dataset, es) + suffix
    filling_labels_filename = commons.data_filename(
        filling_labels_filename, es, fold)

    suffix = commons.testing_suffix
    testing_features_filename = commons.features_name(dataset, es) + suffix
    testing_features_filename = commons.data_filename(
        testing_features_filename, es, fold)
    testing_labels_filename = commons.labels_name(dataset, es) + suffix
    testing_labels_filename = commons.data_filename(
        testing_labels_filename, es, fold)

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)
    return filling_features, filling_labels, testing_features, testing_labels

def match_labels(features, labels, half=False):
    right_features = []
    right_labels = []
    used_idx = set()
    last = 0
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    # Assuming ten clases on each dataset.
    midx = round(len(labels[left_ds]) * 4.0 / 9.0)
    matching_labels = labels[left_ds][:midx] if half else labels[left_ds]
    counter = 0
    print('Matching:')
    for left_label in matching_labels:
        while last in used_idx:
            last += 1
        i = last
        found = False
        for right_feat, right_lab in zip(features[right_ds][i:], labels[right_ds][i:]):
            if (i not in used_idx) and (left_label == right_lab):
                used_idx.add(i)
                right_features.append(right_feat)
                right_labels.append(right_lab)
                found = True
                break
            i += 1
        if not found:
            break
        counter += 1
        commons.print_counter(counter, 1000, 100, symbol='-')
    print(' end')
    if half:
        i = 0
        for right_feat, right_lab in zip(features[right_ds], labels[right_ds]):
            if i not in used_idx:
                right_features.append(right_feat)
                right_labels.append(right_lab)
            i += 1
    n = len(right_features)
    features[left_ds] = features[left_ds][:n]
    labels[left_ds] = labels[left_ds][:n]
    features[right_ds] = np.array(right_features, dtype=int)
    labels[right_ds] = np.array(right_labels, dtype=int)


def describe(features, labels):
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    left_n = len(labels[left_ds])
    right_n = len(labels[right_ds])
    print(f'Elements in left dataset: {left_n}')
    print(f'Elements in right dataset: {right_n}')
    minimum = left_n if left_n < right_n else right_n
    matching = 0
    left_counts = np.zeros((commons.n_labels), dtype=int)
    right_counts = np.zeros((commons.n_labels), dtype=int)
    for i in range(minimum):
        left_label = labels[left_ds][i]
        right_label = labels[right_ds][i]
        left_counts[left_label] += 1
        right_counts[right_label] += 1
        matching += (left_label == right_label)
    print(f'Matching labels: {matching}')
    print(f'Unmatching labels: {minimum - matching}')
    print(f'Left labels counts: {left_counts}')
    print(f'Right labels counts: {right_counts}')


def show_weights_stats(weights):
    w = {}
    conds = ['TP', 'FN', 'FP', 'TN']
    for c in conds:
        if len(weights[c]) == 0:
            w[c] = (0.0, 0.0)
        else:
            mean = np.mean(weights[c])
            stdv = np.std(weights[c])
            w[c] = (mean, stdv)
    print(f'Weights: {w}')


def freqs_to_values(freqs):
    xs = []
    for v, f in enumerate(freqs):
        for _ in range(f):
            xs.append(v)
    random.shuffle(xs)
    return xs


def normality_test(relation):
    ps = []
    for column in relation:
        xs = freqs_to_values(column)
        shapiro_test = scipy.stats.shapiro(xs)
        ps.append(shapiro_test.pvalue)
    return np.mean(ps), np.std(ps)


def statistics_per_fold(dataset, es, fold):
    filling_features, filling_labels, \
        testing_features, testing_labels = features_per_fold(dataset, es, fold)
    print(f'Calculating statistics for fold {fold}')
    return stats_measures(filling_features, filling_labels,
                          testing_features, testing_labels)


def distances_per_fold(dataset, es, fold):
    filling_features, filling_labels, \
        testing_features, testing_labels = features_per_fold(dataset, es, fold)

    print(f'Calculating distances for fold {fold}')
    means, stdvs = distance_matrices(filling_features, filling_labels,
                                     testing_features, testing_labels)
    return means, stdvs


def recognize_by_memory(eam, tef_rounded, tel, msize, minimum, maximum, classifier):
    data = []
    labels = []
    confrix = np.zeros(
        (commons.n_labels, commons.n_labels+1), dtype='int')
    behaviour = np.zeros(commons.n_behaviours, dtype=np.float64)
    unknown = 0
    for features, label in zip(tef_rounded, tel):
        memory, recognized, _ = eam.recall(features)
        if recognized:
            mem = rsize_recall(memory, msize, minimum, maximum)
            data.append(mem)
            labels.append(label)
        else:
            unknown += 1
            confrix[label, commons.n_labels] += 1
    if len(data) > 0:
        data = np.array(data)
        predictions = np.argmax(classifier.predict(data), axis=1)
        for correct, prediction in zip(labels, predictions):
            # For calculation of per memory precision and recall
            confrix[correct, prediction] += 1
    behaviour[commons.no_response_idx] = unknown
    behaviour[commons.correct_response_idx] = \
        np.sum([confrix[i, i] for i in range(commons.n_labels)])

    behaviour[commons.no_correct_response_idx] = \
        len(tel) - unknown - behaviour[commons.correct_response_idx]
    print(f'Unknown elements: {unknown}')
    print(f'Confusion matrix:\n{confrix}')
    print(f'Behaviour: {behaviour}')
    return confrix, behaviour


def recognize_by_hetero_memory(
        hetero_eam: HeteroAssociativeMemory,
        left_eam: AssociativeMemory,
        right_eam: AssociativeMemory,
        tefs, tels):
    confrix = np.zeros((2, 2), dtype=int)
    weights = {'TP': [], 'FN': [], 'FP': [], 'TN': []}
    print('Recognizing by hetero memory')
    counter = 0
    for left_feat, left_lab, right_feat, right_lab \
            in zip(tefs[commons.left_dataset], tels[commons.left_dataset],
                   tefs[commons.right_dataset], tels[commons.right_dataset]):
        _, left_weights = left_eam.recog_weights(left_feat)
        _, right_weights = right_eam.recog_weights(right_feat)
        recognized, weight = hetero_eam.recognize(
            left_feat, right_feat, left_weights, right_weights)
        if recognized:
            if left_lab == right_lab:
                confrix[TP] += 1
                weights['TP'].append(weight)
            else:
                confrix[FP] += 1
                weights['FP'].append(weight)
        else:
            if left_lab == right_lab:
                confrix[FN] += 1
                weights['FN'].append(weight)
            else:
                confrix[TN] += 1
                weights['TN'].append(weight)
        counter += 1
        commons.print_counter(counter, 10000, 1000, symbol='*')
    print(' end')
    show_weights_stats(weights)
    print(f'Confusion matrix:\n{confrix}')
    return confrix


def recall_by_hetero_memory(remembered_dataset, recall,
        eam_origin: AssociativeMemory,
        eam_destination: AssociativeMemory,
        classifier, testing_features, testing_labels,
        msize, mfill, minimum, maximum, mean_weight):
    gc.collect()
    # Each row is a correct label and each column is the prediction, including
    # no recognition.
    confrix = np.zeros(
        (commons.n_labels, commons.n_labels+1), dtype='int')
    behaviour = np.zeros(commons.n_behaviours, dtype=int)
    memories = []
    correct = []
    mem_weights = []
    unknown = 0
    unknown_weights = []
    stats = []
    print('Remembering ', end='')
    counter = 0
    counter_name = commons.set_counter()
    for features, label in zip(testing_features, testing_labels):
        recognized, weights = eam_origin.recog_weights(features)
        if recognized:
            # Recalling using weights.
            memory, recognized, weight, relation, s = recall(features, weights, label)
            if recognized:
                stats.append(s)
                memories.append(memory)
                correct.append(label)
                mem_weights.append(weight)
                if random.randrange(100) == 0:
                    prefix = 'projection-' + remembered_dataset + \
                        '-fill_' + str(int(mfill)).zfill(3) + \
                        '-lbl_' + str(label).zfill(3)
                    plot_relation(relation, prefix)
            else:
                unknown += 1
                confrix[label, commons.n_labels] += 1
                unknown_weights.append(weight)
        else:
            unknown += 1
            confrix[label, commons.n_labels] += 1
        counter += 1
        commons.print_counter(counter, 1000, 100, symbol='+',
            prefix = f'(Recognized : {len(memories)})', name = counter_name)
    print(' done')
    correct_weights = []
    incorrect_weights = []
    correct_stats = []
    incorrect_stats = []
    print('Validating ', end='')
    if len(memories) > 0:
        memories = rsize_recall(np.array(memories), msize, minimum, maximum)
        predictions = np.argmax(classifier.predict(memories), axis=1)
        for label, prediction, weight, s in zip(correct, predictions, mem_weights, stats):
            # For calculation of per memory precision and recall
            confrix[label, prediction] += 1
            if label == prediction:
                correct_weights.append(weight)
                correct_stats.append(s)
            else:
                incorrect_weights.append(weight)
                incorrect_stats.append(s)
    print(' done')
    print(' end')
    behaviour[commons.no_response_idx] = unknown
    behaviour[commons.correct_response_idx] = \
        np.sum([confrix[i, i] for i in range(commons.n_labels)])
    behaviour[commons.no_correct_response_idx] = \
        len(testing_labels) - unknown - \
        behaviour[commons.correct_response_idx]
    print(f'Confusion matrix:\n{confrix}')
    print(f'Behaviour: nr = {behaviour[commons.no_response_idx]}, ' +
          f'ir = {behaviour[commons.no_correct_response_idx]}, ' +
          f'cr = {behaviour[commons.correct_response_idx]}')
    unknown_weights_mean = 0.0 if len(unknown_weights) == 0 \
        else np.mean(unknown_weights/mean_weight)
    unknown_weights_stdv = 0.0 if len(unknown_weights) == 0 \
        else np.std(unknown_weights/mean_weight)
    incorrect_weights_mean = 0.0 if len(incorrect_weights) == 0 \
        else np.mean(incorrect_weights/mean_weight)
    incorrect_weights_stdv = 0.0 if len(incorrect_weights) == 0 \
        else np.std(incorrect_weights/mean_weight)
    correct_weights_mean = 0.0 if len(correct_weights) == 0 \
        else np.mean(correct_weights/mean_weight)
    correct_weights_stdv = 0.0 if len(correct_weights) == 0 \
        else np.std(correct_weights/mean_weight)
    print(f'Mean weight: {mean_weight}')
    print(f'Weights: correct = ({correct_weights_mean}, {correct_weights_stdv}), ' + 
        f'incorrect = ({incorrect_weights_mean}, {incorrect_weights_stdv}), ' +
          f'unknown = ({unknown_weights_mean}, {unknown_weights_stdv})')
    
    if len(stats) > 0:
        stats = np.array(stats)
        stats_mean = np.mean(stats, axis=0)
        stats_stdv = np.std(stats, axis=0)
        stats_skew = scipy.stats.skew(stats)
        stats_kurt = scipy.stats.kurtosis(stats)
        print(f'Stats: mean = {stats_mean}, stdv = {stats_stdv}, ' +
                f'skew = {stats_skew}, kurt = {stats_kurt}.')
    else:
        print(f'Stats not available')
    if len(correct_stats) > 0:
        correct_stats = np.array(correct_stats)
        correct_stats_mean = np.mean(correct_stats, axis=0)
        correct_stats_stdv = np.std(correct_stats, axis=0)
        correct_stats_skew = scipy.stats.skew(correct_stats)
        correct_stats_kurt = scipy.stats.kurtosis(correct_stats)
        print(f'Stats of correct: mean = {correct_stats_mean}, stdev = {correct_stats_stdv}, ' + 
                f'skew = {correct_stats_skew}, kurt = {correct_stats_kurt}.')
    else:
        print('Stats of correct not available')
    if len(incorrect_stats) > 0:
        incorrect_stats = np.array(incorrect_stats)
        incorrect_stats_mean = np.mean(incorrect_stats, axis=0)
        incorrect_stats_stdv = np.std(incorrect_stats, axis=0)
        incorrect_stats_skew = scipy.stats.skew(incorrect_stats)
        incorrect_stats_kurt = scipy.stats.kurtosis(incorrect_stats)
        print(f'Stats of incorrect: mean = {incorrect_stats_mean}, stdev = {incorrect_stats_stdv}, ' + 
                f'skew = {incorrect_stats_skew}, kurt = {incorrect_stats_kurt}.')
    else:
        print('Stats of incorrect not available')
    return confrix, behaviour, memories

def check_hetero_memory(remembered_dataset,
            recall, recognize, inverted, classifier,
            testing_features, testing_cues, testing_labels, fully,
            msize, mfill, minimum, maximum, mean_weight):
    # Each row is a correct label and each column is the prediction, including
    # no recognition.
    gc.collect()
    confrix = np.zeros(
        (commons.n_labels, commons.n_labels+1), dtype='int')
    behaviour = np.zeros(commons.n_behaviours, dtype=int)
    memories = []
    correct_labels = []
    recog_weights = []
    homo_unknown = 0
    hetero_unknown = 0
    homo_unknown_weights = []
    hetero_unknown_weights = []
    recog_not_recall = 0
    recall_not_recog = 0
    print(f'Features shape: {testing_features.shape}')
    print(f'Cues shape: {testing_cues.shape}')
    print('Checking... ', end='')
    cols = commons.datasets_to_domains[remembered_dataset]
    rows = commons.datasets_to_codomains[remembered_dataset]
    counter = 0
    counter_name = commons.set_counter()
    for features, cue, label in zip(testing_features, testing_cues, testing_labels):
        rel_recognized, _ = recognize(cue, features) if inverted else recognize(features, cue) 
        _, recognized, weight, relation, _, _, _ = recall(features)
        recog_not_recall += rel_recognized and not recognized
        recall_not_recog += recognized and not rel_recognized
        if recognized:
            homo = AssociativeMemory(cols, rows, relation = relation)
            candidates = [c[0] for c in zip(testing_cues, testing_labels)
                                if c[1] == label] \
                if fully else [cue]
            for testing_cue in candidates:
                memory, recognized, weight = homo.recall(testing_cue)
                if recognized:
                    memories.append(memory)
                    correct_labels.append(label)
                    recog_weights.append(weight)
                else:
                    homo_unknown += 1
                    homo_unknown_weights.append(weight)
        else:
            hetero_unknown += 1
            confrix[label, commons.n_labels] += 1
            hetero_unknown_weights.append(weight)
        counter += 1
        commons.print_counter(counter, 100, 10, symbol='+',
            prefix = f'(Recognized : {len(memories)})', name = counter_name)

    print(' done.')
    print(f'Not recognized by homo memory: {homo_unknown}')
    print(f'Not recognized by hetero memory: {hetero_unknown}')
    print(f'Recognized but not recalled: {recog_not_recall}')
    print(f'Recalled but not recognized: {recall_not_recog}')
    correct_weights = []
    incorrect_weights = []
    print('Validating... ', end='')
    if len(memories) > 0:
        memories = rsize_recall(np.array(memories), msize, minimum, maximum)
        predictions = np.argmax(classifier.predict(memories), axis=1)
        for label, prediction, weight in zip(correct_labels, predictions, recog_weights):
            # For calculation of per memory precision and recall
            confrix[label, prediction] += 1
            if label == prediction:
                correct_weights.append(weight)
            else:
                incorrect_weights.append(weight)
    print(' done.')
    print(' end.')
    behaviour[commons.no_response_idx] = hetero_unknown + homo_unknown
    behaviour[commons.correct_response_idx] = \
        np.sum([confrix[i, i] for i in range(commons.n_labels)])
    behaviour[commons.no_correct_response_idx] = \
        len(memories) - behaviour[commons.correct_response_idx]
    print(f'Confusion matrix:\n{confrix}')
    print(f'Behaviour: nr = {behaviour[commons.no_response_idx]}, ' +
          f'ir = {behaviour[commons.no_correct_response_idx]}, ' +
          f'cr = {behaviour[commons.correct_response_idx]}')
    incorrect_weights_mean = 0.0 if len(incorrect_weights) == 0 \
        else np.mean(incorrect_weights/mean_weight)
    incorrect_weights_stdv = 0.0 if len(incorrect_weights) == 0 \
        else np.std(incorrect_weights/mean_weight)
    correct_weights_mean = 0.0 if len(correct_weights) == 0 \
        else np.mean(correct_weights/mean_weight)
    correct_weights_stdv = 0.0 if len(correct_weights) == 0 \
        else np.std(correct_weights/mean_weight)
    print(f'Mean weight: {mean_weight}')
    print(f'Weights: correct = ({correct_weights_mean}, {correct_weights_stdv}), ' + 
        f'incorrect = ({incorrect_weights_mean}, {incorrect_weights_stdv})')
    return confrix, behaviour, memories

def remember_by_hetero_memory(eam: HeteroAssociativeMemory,
                              left_eam: AssociativeMemory, right_eam: AssociativeMemory,
                              left_classifier, right_classifier,
                              testing_features, testing_labels,
                              min_maxs, percent, es, fold):
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    rows = commons.codomains()
    confrixes = []
    behaviours = []
    mean_weight = eam.mean
    print('Remembering from left by hetero memory')
    minimum, maximum = min_maxs[right_ds]
    confrix, behaviour, memories = recall_by_hetero_memory(right_ds,
            eam.recall_from_left, left_eam, right_eam, right_classifier,
            testing_features[left_ds], testing_labels[right_ds],
            rows[right_ds], percent, minimum, maximum, mean_weight)
    confrixes.append(confrix)
    behaviours.append(behaviour)
    prefix = commons.memories_name(left_ds, es)
    prefix += commons.int_suffix(percent, 'fll')
    filename = commons.data_filename(prefix, es, fold)
    np.save(filename, memories)
    print('Remembering from right by hetero memory')
    minimum, maximum = min_maxs[left_ds]
    confrix, behaviour, memories = recall_by_hetero_memory(left_ds,
            eam.recall_from_right, right_eam, left_eam, left_classifier,
            testing_features[right_ds], testing_labels[left_ds],
            rows[left_ds], percent, minimum, maximum, mean_weight)
    confrixes.append(confrix)
    behaviours.append(behaviour)
    prefix = commons.memories_name(right_ds, es)
    prefix += commons.int_suffix(percent, 'fll')
    filename = commons.data_filename(prefix, es, fold)
    np.save(filename, memories)
    # confrixes has three dimensions: datasets, correct label, prediction.
    confrixes = np.array(confrixes, dtype=int)
    # behaviours has two dimensions: datasets, behaviours.
    behaviours = np.array(behaviours, dtype=int)
    return confrixes, behaviours

def check_consistency_hetero_memory(
        eam, left_classifier, right_classifier,
        testing_features, testing_labels, fully, min_maxs, percent, es, fold):
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    rows = commons.codomains()
    confrixes = []
    behaviours = []
    mean_weight = eam.mean
    print('Checking consistency of hetero memory from left')
    minimum, maximum = min_maxs[right_ds]
    confrix, behaviour, memories = check_hetero_memory(right_ds,
            eam.recall_from_left, eam.recognize, False, right_classifier,
            testing_features[left_ds], testing_features[right_ds], testing_labels[right_ds], fully,
            rows[right_ds], percent, minimum, maximum, mean_weight)
    confrixes.append(confrix)
    behaviours.append(behaviour)
    prefix = commons.memories_name(left_ds, es)
    prefix += commons.int_suffix(percent, 'fll')
    filename = commons.data_filename(prefix, es, fold)
    np.save(filename, memories)
    print('Checking consistency of hetero memory from right')
    minimum, maximum = min_maxs[left_ds]
    confrix, behaviour, memories = check_hetero_memory(left_ds,
            eam.recall_from_right, eam.recognize, True, left_classifier,
            testing_features[right_ds], testing_features[left_ds], testing_labels[left_ds], fully,
            rows[left_ds], percent, minimum, maximum, mean_weight)
    confrixes.append(confrix)
    behaviours.append(behaviour)
    prefix = commons.memories_name(right_ds, es)
    prefix += commons.int_suffix(percent, 'fll')
    filename = commons.data_filename(prefix, es, fold)
    np.save(filename, memories)
    # confrixes has three dimensions: datasets, correct label, prediction.
    confrixes = np.array(confrixes, dtype=int)
    # behaviours has two dimensions: datasets, behaviours.
    behaviours = np.array(behaviours, dtype=int)
    return confrixes, behaviours

def optimum_indexes(precisions, recalls):
    f1s = []
    i = 0
    for p, r in zip(precisions, recalls):
        f1 = 0 if (r+p) == 0 else 2*(r*p)/(r+p)
        f1s.append((f1, i))
        i += 1
    f1s.sort(reverse=True, key=lambda tuple: tuple[0])
    return [t[1] for t in f1s[:commons.n_best_memory_sizes]]


def get_ams_results(
        midx, msize, domain,
        filling_features, testing_features,
        filling_labels, testing_labels, classifier, es):
    # Round the values
    min_value, max_value = get_min_max(filling_features)

    trf_rounded = msize_features(filling_features, msize, min_value, max_value)
    tef_rounded = msize_features(testing_features, msize, min_value, max_value)
    behaviour = np.zeros(commons.n_behaviours, dtype=np.float64)

    # Create the memory using default parameters.
    params = commons.ExperimentSettings()
    eam = AssociativeMemory(domain, msize, params)

    # Registrate filling data.
    for features in trf_rounded:
        eam.register(features)

    # Recognize test data.
    confrix, behaviour = recognize_by_memory(
        eam, tef_rounded, testing_labels, msize, min_value, max_value, classifier)
    responses = len(testing_labels) - behaviour[commons.no_response_idx]
    precision = behaviour[commons.correct_response_idx]/float(responses)
    recall = behaviour[commons.correct_response_idx] / \
        float(len(testing_labels))
    behaviour[commons.precision_idx] = precision
    behaviour[commons.recall_idx] = recall
    return midx, eam.entropy, behaviour, confrix


def statistics(dataset, es):
    list_results = []
    for fold in range(commons.n_folds):
        results = statistics_per_fold(dataset, es, fold)
        print(f'Results: {results}')
        list_results.append(results)
    means = []
    stdvs = []
    for mean, stdv in list_results:
        means.append(mean)
        stdvs.append(stdv)
    means = np.concatenate(means, axis=1)
    stdvs = np.concatenate(stdvs, axis=1)
    data = [means, stdvs]
    suffixes = ['-means', '-stdvs']
    for d, suffix in zip(data, suffixes):
        print(f'Shape{suffix[0]},{suffix[1]}: {d.shape}')
        filename = commons.fstats_name(dataset, es)
        filename += suffix
        filename = commons.csv_filename(filename, es)
        np.savetxt(filename, d, delimiter=',')


def distances(dataset, es):
    distance_means = []
    distance_stdvs = []
    for fold in range(commons.n_folds):
        mean, stdv = distances_per_fold(dataset, es, fold)
        distance_means.append(mean)
        distance_stdvs.append(stdv)
        plot_distances(mean, f'distances_{dataset}', es, fold)
    distance_means = np.concatenate(distance_means, axis=1)
    distance_stdvs = np.concatenate(distance_stdvs, axis=1)
    data = [distance_means, distance_stdvs]
    suffixes = ['-means', '-stdvs']
    for d, suffix in zip(data, suffixes):
        print(f'Shape{suffix[0]},{suffix[1]}: {d.shape}')
        filename = commons.distance_name(dataset, es)
        filename += suffix
        filename = commons.csv_filename(filename, es)
        np.savetxt(filename, d, delimiter=',')


def test_memory_sizes(dataset, es):
    domain = commons.domain(dataset)
    all_entropies = []
    precision = []
    recall = []
    all_confrixes = []
    no_response = []
    no_correct_response = []
    correct_response = []

    print(f'Testing the memory of {dataset}')
    model_prefix = commons.model_name(dataset, es)
    for fold in range(commons.n_folds):
        gc.collect()
        filename = commons.classifier_filename(model_prefix, es, fold)
        classifier = tf.keras.models.load_model(filename)
        print(f'Fold: {fold}')
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(
            dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold)
        filling_labels_filename = commons.labels_name(dataset, es) + suffix
        filling_labels_filename = commons.data_filename(
            filling_labels_filename, es, fold)

        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(
            dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold)
        testing_labels_filename = commons.labels_name(dataset, es) + suffix
        testing_labels_filename = commons.data_filename(
            testing_labels_filename, es, fold)

        filling_features = np.load(filling_features_filename)
        filling_labels = np.load(filling_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)
        validating_network_data(
            filling_features, filling_labels, classifier, dataset, 'filling data')
        validating_network_data(
            testing_features, testing_labels, classifier, dataset, 'testing data')
        behaviours = np.zeros(
            (len(commons.memory_sizes), commons.n_behaviours))
        measures = []
        confrixes = []
        entropies = []
        for midx, msize in enumerate(commons.memory_sizes):
            print(f'Memory size: {msize}')
            results = get_ams_results(midx, msize, domain,
                                      filling_features, testing_features,
                                      filling_labels, testing_labels, classifier, es)
            measures.append(results)
        for midx, entropy, behaviour, confrix in measures:
            entropies.append(entropy)
            behaviours[midx, :] = behaviour
            confrixes.append(confrix)

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        all_entropies.append(entropies)

        # Average precision and recall as percentage
        precision.append(behaviours[:, commons.precision_idx]*100)
        recall.append(behaviours[:, commons.recall_idx]*100)

        all_confrixes.append(np.array(confrixes))
        no_response.append(behaviours[:, commons.no_response_idx])
        no_correct_response.append(
            behaviours[:, commons.no_correct_response_idx])
        correct_response.append(behaviours[:, commons.correct_response_idx])

    # Every row is training fold, and every column is a memory size.
    all_entropies = np.array(all_entropies)
    precision = np.array(precision)
    recall = np.array(recall)
    all_confrixes = np.array(all_confrixes)

    average_entropy = np.mean(all_entropies, axis=0)
    average_precision = np.mean(precision, axis=0)
    stdev_precision = np.std(precision, axis=0)
    average_recall = np.mean(recall, axis=0)
    stdev_recall = np.std(recall, axis=0)
    average_confrixes = np.mean(all_confrixes, axis=0)

    no_response = np.array(no_response)
    no_correct_response = np.array(no_correct_response)
    correct_response = np.array(correct_response)
    mean_no_response = np.mean(no_response, axis=0)
    stdv_no_response = np.std(no_response, axis=0)
    mean_no_correct_response = np.mean(no_correct_response, axis=0)
    stdv_no_correct_response = np.std(no_correct_response, axis=0)
    mean_correct_response = np.mean(correct_response, axis=0)
    stdv_correct_response = np.std(correct_response, axis=0)
    best_memory_idx = optimum_indexes(average_precision, average_recall)
    best_memory_sizes = [commons.memory_sizes[i] for i in best_memory_idx]
    mean_behaviours = \
        [mean_no_response, mean_no_correct_response, mean_correct_response]
    stdv_behaviours = \
        [stdv_no_response, stdv_no_correct_response, stdv_correct_response]

    np.savetxt(commons.csv_filename(
        'memory_precision-' + dataset, es), precision, delimiter=',')
    np.savetxt(commons.csv_filename(
        'memory_recall-' + dataset, es), recall, delimiter=',')
    np.savetxt(commons.csv_filename(
        'memory_entropy-' + dataset, es), all_entropies, delimiter=',')
    np.savetxt(commons.csv_filename('mean_behaviours-' + dataset, es),
               mean_behaviours, delimiter=',')
    np.savetxt(commons.csv_filename('stdv_behaviours-' + dataset, es),
               stdv_behaviours, delimiter=',')
    np.save(commons.data_filename(
        'memory_confrixes-' + dataset, es), average_confrixes)
    np.save(commons.data_filename('behaviours-' + dataset, es), behaviours)
    plot_prerec_graph(average_precision, average_recall, average_entropy,
                   stdev_precision, stdev_recall, dataset, es, prefix='homo_msizes-')
    plot_behs_graph(mean_no_response, mean_no_correct_response,
                    mean_correct_response, dataset, es, prefix='homo_msizes-')
    print('Memory size evaluation completed!')
    return best_memory_sizes


def test_filling_percent(
        eam, msize, min_value, max_value,
        trf, tef, tel, percent, classifier):
    # Registrate filling data.
    for features in trf:
        eam.register(features)
    print(f'Filling of memories done at {percent}%')
    _, behaviour = recognize_by_memory(
        eam, tef, tel, msize, min_value, max_value, classifier)
    responses = len(tel) - behaviour[commons.no_response_idx]
    precision = behaviour[commons.correct_response_idx]/float(responses)
    recall = behaviour[commons.correct_response_idx]/float(len(tel))
    behaviour[commons.precision_idx] = precision
    behaviour[commons.recall_idx] = recall
    return behaviour, eam.entropy


def test_hetero_filling_percent(
        hetero_eam: HeteroAssociativeMemory,
        left_eam: AssociativeMemory,
        right_eam: AssociativeMemory,
        trfs, tefs, tels, percent):
    # Register filling data.
    print('Filling hetero memory')
    counter = 0
    for left_feat, right_feat \
            in zip(trfs[commons.left_dataset], trfs[commons.right_dataset]):
        hetero_eam.register(left_feat, right_feat)
        counter += 1
        commons.print_counter(counter, 1000, 100)
    print(' end')
    print(f'Filling of memories done at {percent}%')
    print(f'Memory full at {100*hetero_eam.fullness}%')
    confrix = recognize_by_hetero_memory(
        hetero_eam, left_eam, right_eam, tefs, tels)
    return confrix, hetero_eam.entropy


def hetero_remember_percent(
        eam: HeteroAssociativeMemory,
        left_eam: AssociativeMemory,
        right_eam: AssociativeMemory,
        left_classifier, right_classifier,
        filling_features, testing_features, testing_labels,
        min_maxs, percent, es, fold):
    # Register filling data.
    print('Filling hetero memory')
    counter = 0
    for left_feat, right_feat \
            in zip(filling_features[commons.left_dataset],
                   filling_features[commons.right_dataset]):
        eam.register(left_feat, right_feat)
        counter += 1
        commons.print_counter(counter, 1000, 100)
    print(' end')
    print(f'Filling of memories done at {percent}%')
    print(f'Memory full at {100*eam.fullness}%')
    confrixes, behaviours = remember_by_hetero_memory(
        eam, left_eam, right_eam, left_classifier, right_classifier,
        testing_features, testing_labels, min_maxs, percent, es, fold)
    return confrixes, behaviours, eam.entropy

def hetero_check_consistency_percent(
        eam: HeteroAssociativeMemory, left_classifier, right_classifier,
        filling_features, filling_labels, testing_features, testing_labels, fully, 
        min_maxs, percent, es, fold):
    # Register filling data in the hetero-associative memory.
    print('Filling hetero memory')
    counter = 0
    for left_feat, right_feat \
            in zip(filling_features[commons.left_dataset],
                   filling_features[commons.right_dataset]):
        eam.register(left_feat, right_feat)
        counter += 1
        commons.print_counter(counter, 1000, 100)
    print(' end')
    print(f'Filling of memories done at {percent}%')
    # Check consistency, either using filling or testing data.
    confrixes, behaviours = check_consistency_hetero_memory(eam,
            left_classifier, right_classifier,
            testing_features, testing_labels, fully,
            min_maxs, percent, es, fold)
    return confrixes, behaviours, eam.entropy
    
def test_filling_per_fold(mem_size, domain, dataset, es, fold):
    # Create the required associative memories using default parameters.
    params = commons.ExperimentSettings()
    eam = AssociativeMemory(domain, mem_size, params)
    model_prefix = commons.model_name(dataset, es)
    filename = commons.classifier_filename(model_prefix, es, fold)
    classifier = tf.keras.models.load_model(filename)

    suffix = commons.filling_suffix
    filling_features_filename = commons.features_name(dataset, es) + suffix
    filling_features_filename = commons.data_filename(
        filling_features_filename, es, fold)
    filling_labels_filename = commons.labels_name(dataset, es) + suffix
    filling_labels_filename = commons.data_filename(
        filling_labels_filename, es, fold)

    suffix = commons.testing_suffix
    testing_features_filename = commons.features_name(dataset, es) + suffix
    testing_features_filename = commons.data_filename(
        testing_features_filename, es, fold)
    testing_labels_filename = commons.labels_name(dataset, es) + suffix
    testing_labels_filename = commons.data_filename(
        testing_labels_filename, es, fold)

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    min_value, max_value = get_min_max(filling_features)
    filling_features = msize_features(
        filling_features, mem_size, min_value, max_value)
    testing_features = msize_features(
        testing_features, mem_size, min_value, max_value)

    total = len(filling_labels)
    percents = np.array(commons.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []

    start = 0
    for percent, end in zip(percents, steps):
        features = filling_features[start:end]
        print(f'Filling from {start} to {end}.')
        behaviour, entropy = \
            test_filling_percent(eam, mem_size,
                                 min_value, max_value, features,
                                 testing_features, testing_labels, percent, classifier)
        # A list of tuples (position, label, features)
        # fold_recalls += recalls
        # An array with average entropy per step.
        fold_entropies.append(entropy)
        # Arrays with precision, and recall.
        fold_precision.append(behaviour[commons.precision_idx])
        fold_recall.append(behaviour[commons.recall_idx])
        start = end
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_recall = np.array(fold_recall)
    print(f'Filling test completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_recall


def test_hetero_filling_per_fold(es, fold):
    # Create the required associative memories.
    domains = commons.domains()
    rows = commons.codomains()
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    params = commons.ExperimentSettings()
    left_eam = AssociativeMemory(domains[left_ds], rows[left_ds], params)
    right_eam = AssociativeMemory(domains[right_ds], rows[right_ds], params)
    hetero_eam = HeteroAssociativeMemory(domains[left_ds], domains[right_ds],
        rows[left_ds], rows[right_ds], es, fold)
    filling_features = {}
    filling_labels = {}
    testing_features = {}
    testing_labels = {}
    for dataset in commons.datasets:
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(
            dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold)
        filling_labels_filename = commons.labels_name(dataset, es) + suffix
        filling_labels_filename = commons.data_filename(
            filling_labels_filename, es, fold)

        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(
            dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold)
        testing_labels_filename = commons.labels_name(dataset, es) + suffix
        testing_labels_filename = commons.data_filename(
            testing_labels_filename, es, fold)

        filling_labels[dataset] = np.load(filling_labels_filename)
        testing_labels[dataset] = np.load(testing_labels_filename)
        f_features = np.load(filling_features_filename)
        t_features = np.load(testing_features_filename)
        min_value, max_value = get_min_max(f_features)
        filling_features[dataset] = msize_features(
            f_features, rows[dataset], min_value, max_value)
        testing_features[dataset] = msize_features(
            t_features, rows[dataset], min_value, max_value)
    match_labels(filling_features, filling_labels)
    describe(filling_features, filling_labels)
    match_labels(testing_features, testing_labels, half=True)
    describe(testing_features, testing_labels)
    for f in filling_features[left_ds]:
        left_eam.register(f)
    for f in filling_features[right_ds]:
        right_eam.register(f)
    total = len(filling_features[left_ds])
    print(f'Filling hetero-associative memory with a total of {total} pairs.')
    percents = np.array(commons.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_accuracy = []

    start = 0
    for percent, end in zip(percents, steps):
        features = {}
        features[left_ds] = filling_features[left_ds][start:end]
        features[right_ds] = filling_features[right_ds][start:end]
        print(f'Filling from {start} to {end}.')
        confrix, entropy = \
            test_hetero_filling_percent(
                hetero_eam, left_eam, right_eam,
                features, testing_features, testing_labels, percent)
        # An array with average entropy per step.
        fold_entropies.append(entropy)
        # Arrays with precision, and recall.
        positives = confrix[TP]+confrix[FP]
        fold_precision.append(1.0 if positives == 0 else confrix[TP]/positives)
        fold_recall.append(confrix[TP]/(confrix[TP]+confrix[FN]))
        fold_accuracy.append((confrix[TP]+confrix[TN])/np.sum(confrix))
        start = end
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_recall = np.array(fold_recall)
    fold_accuracy = np.array(fold_accuracy)
    print(
        f'Filling test of hetero-associative memory completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_recall, fold_accuracy


def hetero_remember_per_fold(es, fold):
    # Create the required associative memories.
    domains = commons.domains()
    rows = commons.codomains()
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    params = commons.ExperimentSettings()
    left_eam = AssociativeMemory(domains[left_ds], rows[left_ds], params)
    right_eam = AssociativeMemory(domains[right_ds], rows[right_ds], params)

    # Retrieve the classifiers.
    model_prefix = commons.model_name(left_ds, es)
    filename = commons.classifier_filename(model_prefix, es, fold)
    left_classifier = tf.keras.models.load_model(filename)
    model_prefix = commons.model_name(right_ds, es)
    filename = commons.classifier_filename(model_prefix, es, fold)
    right_classifier = tf.keras.models.load_model(filename)
    classifiers = {
        left_ds: left_classifier,
        right_ds: right_classifier
    }
    filling_features = {}
    filling_labels = {}
    testing_features = {}
    testing_labels = {}
    filling_prototypes = {}
    min_maxs = {}
    for dataset in commons.datasets:
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(
            dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold)
        # Original tagging of filling data.
        filling_labels_filename = commons.labels_name(dataset, es) + suffix
        filling_labels_filename = commons.data_filename(
            filling_labels_filename, es, fold)
        filling_labels[dataset] = np.load(filling_labels_filename)
        f_features = np.load(filling_features_filename)
        # Tagging of filling data produced by the classifier
        # labels = np.argmax(classifiers[dataset].predict(f_features), axis=1)
        # filling_labels[dataset] = labels
        prototypes = None
        if commons.use_prototypes:
            proto_suffix = suffix + commons.proto_suffix
            proto_filename = commons.features_name(dataset, es) + proto_suffix
            proto_filename = commons.data_filename(proto_filename, es, fold)
            prototypes = np.load(proto_filename)

        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(
            dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold)
        # Original tagging of testing data.
        testing_labels_filename = commons.labels_name(dataset, es) + suffix
        testing_labels_filename = commons.data_filename(
            testing_labels_filename, es, fold)
        testing_labels[dataset] = np.load(testing_labels_filename)
        t_features = np.load(testing_features_filename)
        # Tagging of testing data produced by the classifier
        # labels = np.argmax(classifiers[dataset].predict(t_features), axis=1)
        # testing_labels[dataset] = labels
        validating_network_data(
            f_features, filling_labels[dataset], classifiers[dataset],
            dataset, 'filling data')
        validating_network_data(
            t_features, testing_labels[dataset], classifiers[dataset],
            dataset, 'testing data')
        min_value, max_value = get_min_max(f_features)
        filling_features[dataset] = msize_features(
            f_features, rows[dataset], min_value, max_value)
        filling_prototypes[dataset] = None if prototypes is None \
                else msize_features(prototypes, rows[dataset], min_value, max_value)
        testing_features[dataset] = msize_features(
            t_features, rows[dataset], min_value, max_value)
        min_maxs[dataset] = [min_value, max_value]

    eam = HeteroAssociativeMemory(domains[left_ds], domains[right_ds],
            rows[left_ds], rows[right_ds], es, fold, min_value, max_value,
            [filling_prototypes[left_ds], filling_prototypes[right_ds]])
    
    for f in filling_features[left_ds]:
        left_eam.register(f)
    for f in filling_features[right_ds]:
        right_eam.register(f)
    match_labels(filling_features, filling_labels)
    describe(filling_features, filling_labels)
    match_labels(testing_features, testing_labels)
    total = len(filling_labels[left_ds])
    total_test = len(testing_labels[left_ds])
    top = int(commons.exploration_percent*total_test)
    testing_labels[left_ds] = testing_labels[left_ds][:top]
    testing_features[left_ds] = testing_features[left_ds][:top]
    testing_labels[right_ds] = testing_labels[right_ds][:top]
    testing_features[right_ds] = testing_features[right_ds][:top]
    describe(testing_features, testing_labels)
    print(f'Filling hetero-associative memory with a total of {total} pairs.')
    percents = np.array(commons.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_confrixes = []
    fold_behaviours = []
    start = 0
    for percent, end in zip(percents, steps):
        features = {}
        features[left_ds] = filling_features[left_ds][start:end]
        features[right_ds] = filling_features[right_ds][start:end]
        print(f'Filling from {start} to {end}.')
        confrixes, behaviours, entropy = \
            hetero_remember_percent(
                eam, left_eam, right_eam, left_classifier, right_classifier,
                features, testing_features, testing_labels, min_maxs, percent, es, fold)
        fold_entropies.append(entropy)
        fold_behaviours.append(behaviours)
        fold_confrixes.append(confrixes)
        # Arrays with precision, and recall.
        no_response = behaviours[:, commons.no_response_idx]
        correct = behaviours[:, commons.correct_response_idx]
        incorrect = behaviours[:, commons.no_correct_response_idx]
        with np.errstate(divide='ignore'):
            fold_precision.append(np.where((correct + incorrect) == 0,
                1.0, correct/(correct+incorrect)))
        fold_recall.append(
            behaviours[:, commons.correct_response_idx]/(no_response + correct + incorrect))
        start = end
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.transpose(np.array(fold_precision))
    fold_recall = np.transpose(np.array(fold_recall))
    fold_behaviours = np.transpose(
        np.array(fold_behaviours, dtype=int), axes=(1, 0, 2))
    fold_confrixes = np.transpose(
        np.array(fold_confrixes, dtype=int), axes=(1, 0, 2, 3))
    print(
        f'Filling test of hetero-associative memory completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_recall, fold_confrixes, fold_behaviours

def check_consistency_per_fold(filling, es, fold):
    # Create the required associative memories.
    domains = commons.domains()
    rows = commons.codomains()
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    eam = HeteroAssociativeMemory(domains[left_ds], domains[right_ds],
            rows[left_ds], rows[right_ds], es, fold)

    # Retrieve the classifiers.
    model_prefix = commons.model_name(left_ds, es)
    filename = commons.classifier_filename(model_prefix, es, fold)
    left_classifier = tf.keras.models.load_model(filename)
    model_prefix = commons.model_name(right_ds, es)
    filename = commons.classifier_filename(model_prefix, es, fold)
    right_classifier = tf.keras.models.load_model(filename)
    classifiers = {
        left_ds: left_classifier,
        right_ds: right_classifier
    }
    filling_features = {}
    filling_labels = {}
    testing_features = {}
    testing_labels = {}
    min_maxs = {}
    for dataset in commons.datasets:
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(
            dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold)
        filling_labels_filename = commons.labels_name(dataset, es) + suffix
        filling_labels_filename = commons.data_filename(
            filling_labels_filename, es, fold)

        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(
            dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold)
        testing_labels_filename = commons.labels_name(dataset, es) + suffix
        testing_labels_filename = commons.data_filename(
            testing_labels_filename, es, fold)

        filling_labels[dataset] = np.load(filling_labels_filename)
        testing_labels[dataset] = np.load(testing_labels_filename)
        f_features = np.load(filling_features_filename)
        t_features = np.load(testing_features_filename)
        validating_network_data(
            f_features, filling_labels[dataset], classifiers[dataset],
            dataset, 'filling data')
        validating_network_data(
            t_features, testing_labels[dataset], classifiers[dataset],
            dataset, 'testing data')
        min_value, max_value = get_min_max(f_features)
        filling_features[dataset] = msize_features(
            f_features, rows[dataset], min_value, max_value)
        testing_features[dataset] = msize_features(
            t_features, rows[dataset], min_value, max_value)
        min_maxs[dataset] = [min_value, max_value]

    match_labels(filling_features, filling_labels)
    describe(filling_features, filling_labels)
    match_labels(testing_features, testing_labels)
    describe(testing_features, testing_labels)
    total = len(filling_labels[left_ds])
    percents = np.array(commons.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_confrixes = []
    fold_behaviours = []
    start = 0
    for percent, end in zip(percents, steps):
        features = {}
        labels = {}
        features[left_ds] = filling_features[left_ds][start:end]
        features[right_ds] = filling_features[right_ds][start:end]
        labels[left_ds] = filling_labels[left_ds][start:end]
        labels[right_ds] = filling_labels[right_ds][start:end]
        print(f'Filling from {start} to {end}.')
        if filling:
            testing_features[left_ds] = filling_features[left_ds][:end]
            testing_features[right_ds] = filling_features[right_ds][:end]
            testing_labels[left_ds] = filling_labels[left_ds][:end]
            testing_labels[right_ds] = filling_labels[right_ds][:end]
        confrixes, behaviours, entropy = \
            hetero_check_consistency_percent(
                eam, left_classifier, right_classifier,
                features, labels, testing_features, testing_labels, not filling,
                min_maxs, percent, es, fold)
        fold_entropies.append(entropy)
        fold_behaviours.append(behaviours)
        fold_confrixes.append(confrixes)
        # Arrays with precision, and recall.
        noresponse = behaviours[:, commons.no_response_idx]
        correct = behaviours[:, commons.correct_response_idx]
        incorrect = behaviours[:, commons.no_correct_response_idx]
        fold_precision.append(np.where(correct + incorrect == 0,
            1.0, correct/(correct+incorrect)))
        fold_recall.append(
            behaviours[:,
                commons.correct_response_idx]/(noresponse+correct+incorrect))
        start = end
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.transpose(np.array(fold_precision))
    fold_recall = np.transpose(np.array(fold_recall))
    fold_behaviours = np.transpose(
        np.array(fold_behaviours, dtype=int), axes=(1, 0, 2))
    fold_confrixes = np.transpose(
        np.array(fold_confrixes, dtype=int), axes=(1, 0, 2, 3))
    print(
        f'Filling test of hetero-associative memory completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_recall, fold_confrixes, fold_behaviours

def test_memory_fills(mem_sizes, dataset, es):
    domain = commons.domain(dataset)
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    best_filling_percents = []
    for mem_size in mem_sizes:
        # All entropies, precision, and recall, per size, fold, and fill.
        total_entropies = np.zeros((testing_folds, len(memory_fills)))
        total_precisions = np.zeros((testing_folds, len(memory_fills)))
        total_recalls = np.zeros((testing_folds, len(memory_fills)))
        list_results = []

        for fold in range(testing_folds):
            results = test_filling_per_fold(
                mem_size, domain, dataset, es, fold)
            list_results.append(results)
        for fold, entropies, precisions, recalls in list_results:
            total_precisions[fold] = precisions
            total_recalls[fold] = recalls
            total_entropies[fold] = entropies

        main_avrge_entropies = np.mean(total_entropies, axis=0)
        main_stdev_entropies = np.std(total_entropies, axis=0)
        main_avrge_precisions = np.mean(total_precisions, axis=0)
        main_stdev_precisions = np.std(total_precisions, axis=0)
        main_avrge_recalls = np.mean(total_recalls, axis=0)
        main_stdev_recalls = np.std(total_recalls, axis=0)

        np.savetxt(
            commons.csv_filename(
                'main_average_precision-' + dataset +
                commons.numeric_suffix('sze', mem_size), es),
            main_avrge_precisions, delimiter=',')
        np.savetxt(
            commons.csv_filename(
                'main_average_recall-' + dataset + commons.numeric_suffix('sze', mem_size), es),
            main_avrge_recalls, delimiter=',')
        np.savetxt(
            commons.csv_filename(
                'main_average_entropy-' + dataset + commons.numeric_suffix('sze', mem_size), es),
            main_avrge_entropies, delimiter=',')
        np.savetxt(
            commons.csv_filename(
                'main_stdev_precision-' + dataset + commons.numeric_suffix('sze', mem_size), es),
            main_stdev_precisions, delimiter=',')
        np.savetxt(
            commons.csv_filename(
                'main_stdev_recall-' + dataset + commons.numeric_suffix('sze', mem_size), es),
            main_stdev_recalls, delimiter=',')
        np.savetxt(
            commons.csv_filename(
                'main_stdev_entropy-' + dataset + commons.numeric_suffix('sze', mem_size), es),
            main_stdev_entropies, delimiter=',')

        plot_prerec_graph(main_avrge_precisions*100, main_avrge_recalls*100, main_avrge_entropies,
                       main_stdev_precisions*100, main_stdev_recalls *
                       100, dataset, es,
                       prefix='homo_fills' +
                       commons.numeric_suffix('sze', mem_size) + '-',
                       xlabels=commons.memory_fills, xtitle=_('Percentage of memory corpus'))

        bf_idx = optimum_indexes(
            main_avrge_precisions, main_avrge_recalls)
        best_filling_percents.append(commons.memory_fills[bf_idx[0]])
        print(f'Testing fillings for memory size {mem_size} done.')
    return best_filling_percents


def test_hetero_fills(es):
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    # All entropies, precision, and recall, per size, fold, and fill.
    total_entropies = np.zeros((testing_folds, len(memory_fills)))
    total_precisions = np.zeros((testing_folds, len(memory_fills)))
    total_recalls = np.zeros((testing_folds, len(memory_fills)))
    total_accuracies = np.zeros((testing_folds, len(memory_fills)))
    list_results = []

    for fold in range(testing_folds):
        results = test_hetero_filling_per_fold(es, fold)
        list_results.append(results)
    for fold, entropies, precisions, recalls, accuracies in list_results:
        total_precisions[fold] = precisions
        total_recalls[fold] = recalls
        total_entropies[fold] = entropies
        total_accuracies[fold] = accuracies
    main_avrge_entropies = np.mean(total_entropies, axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_precisions = np.mean(total_precisions, axis=0)
    main_stdev_precisions = np.std(total_precisions, axis=0)
    main_avrge_recalls = np.mean(total_recalls, axis=0)
    main_stdev_recalls = np.std(total_recalls, axis=0)
    main_avrge_accuracies = np.mean(total_accuracies, axis=0)
    main_stdev_accuracies = np.std(total_accuracies, axis=0)

    np.savetxt(
        commons.csv_filename(
            'hetero_average_precision', es),
        main_avrge_precisions, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'hetero_average_recall', es),
        main_avrge_recalls, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'hetero_average_accuracy', es),
        main_avrge_accuracies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'hetero_average_entropy', es),
        main_avrge_entropies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'hetero_stdev_precision', es),
        main_stdev_precisions, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'hetero_stdev_recall', es),
        main_stdev_recalls, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'hetero_stdev_accuracy', es),
        main_stdev_accuracies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'hetero_stdev_entropy', es),
        main_stdev_entropies, delimiter=',')

    prefix = 'hetero_recognize-'
    plot_prerec_graph(100*main_avrge_precisions, 100*main_avrge_recalls, main_avrge_entropies,
                   100*main_stdev_precisions, 100*main_stdev_recalls, 'hetero',
                   es, acc_mean=100*main_avrge_accuracies, acc_std=100*main_stdev_accuracies,
                   prefix=prefix,
                   xlabels=commons.memory_fills, xtitle=_('Percentage of memory corpus'))
    print('Testing fillings for hetero-associative done.')

def validating_network_data(features, labels, classifier, dataset, description):
    predictions = np.argmax(classifier.predict(features), axis=1)
    total = labels.size
    agreements = np.count_nonzero(predictions == labels)
    print(f'Validating coherence between data and network for {description} of {dataset}')
    print(f'Agreement percentage: {100*agreements/total}')

def save_history(history, prefix, es):
    """ Saves the stats of neural networks.

    Neural networks stats may come either as a History object, that includes
    a History.history dictionary with stats, or directly as a dictionary.
    """
    stats = {}
    stats['history'] = []
    for h in history:
        while not isinstance(h, (dict, list)):
            h = h.history
        stats['history'].append(h)
    with open(commons.json_filename(prefix, es), 'w') as outfile:
        json.dump(stats, outfile)


def save_conf_matrix(matrix, dataset, prefix, es, vmax=None):
    plot_confusion_matrix(matrix, range(commons.n_labels), dataset, es, prefix, vmax=vmax)
    fname = prefix + commons.matrix_suffix + '-' + dataset
    filename = commons.data_filename(fname)
    np.save(filename, matrix)


def save_learned_params(mem_sizes, fill_percents, dataset, es):
    name = commons.learn_params_name(dataset, es)
    filename = commons.data_filename(name, es)
    np.save(filename, np.array([mem_sizes, fill_percents], dtype=int))


def remember(es):
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    total_entropies = np.zeros((testing_folds, len(memory_fills)))
    # We are capturing left and right measures.
    total_precisions = np.zeros((testing_folds, 2, len(memory_fills)))
    total_recalls = np.zeros((testing_folds, 2, len(memory_fills)))
    total_accuracies = np.zeros((testing_folds, 2, len(memory_fills)))
    total_confrixes = []
    total_behaviours = []

    for fold in range(testing_folds):
        fold, entropies, precisions, recalls, confrixes, behaviours = \
            hetero_remember_per_fold(es, fold)
        total_precisions[fold] = precisions
        total_recalls[fold] = recalls
        total_entropies[fold] = entropies
        total_confrixes.append(confrixes)
        total_behaviours.append(behaviours)
    total_confrixes = np.array(total_confrixes, dtype=int)
    total_behaviours = np.array(total_behaviours, dtype=int)

    main_avrge_entropies = np.mean(total_entropies, axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_precisions = np.mean(total_precisions, axis=0)
    main_stdev_precisions = np.std(total_precisions, axis=0)
    main_avrge_recalls = np.mean(total_recalls, axis=0)
    main_stdev_recalls = np.std(total_recalls, axis=0)
    main_avrge_accuracies = np.mean(total_accuracies, axis=0)
    main_stdev_accuracies = np.std(total_accuracies, axis=0)
    main_avrge_confrixes = np.mean(total_confrixes, axis=0)
    main_stdev_confrixes = np.std(total_confrixes, axis=0)
    main_avrge_behaviours = np.mean(total_behaviours, axis=0)
    main_stdev_behaviours = np.std(total_behaviours, axis=0)
    np.savetxt(
        commons.csv_filename(
            'remember_average_precision', es),
        main_avrge_precisions, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'remember_average_recall', es),
        main_avrge_recalls, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'remember_average_accuracy', es),
        main_avrge_accuracies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'remember_average_entropy', es),
        main_avrge_entropies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'remember_stdev_precision', es),
        main_stdev_precisions, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'remember_stdev_recall', es),
        main_stdev_recalls, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'remember_stdev_accuracy', es),
        main_stdev_accuracies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'remember_stdev_entropy', es),
        main_stdev_entropies, delimiter=',')
    np.save(commons.data_filename('remember_mean_behaviours', es),
            main_avrge_behaviours)
    np.save(commons.data_filename('remember_stdv_behaviours', es),
            main_stdev_behaviours)
    np.save(commons.data_filename('remember_mean_confrixes', es),
            main_avrge_confrixes)
    np.save(commons.data_filename('remember_stdv_confrixes', es),
            main_stdev_confrixes)

    for i in range(len(commons.datasets)):
        dataset = commons.datasets[i]
        plot_prerec_graph(
            100*main_avrge_precisions[i], 100 *
            main_avrge_recalls[i], main_avrge_entropies,
            100*main_stdev_precisions[i], 100*main_stdev_recalls[i], dataset,
            es, acc_mean=100*main_avrge_accuracies[i], acc_std=100*main_stdev_accuracies[i],
            prefix='hetero_remember-', xlabels=commons.memory_fills,
            xtitle=_('Percentage of memory corpus'))
        mean_no_response = main_avrge_behaviours[i,
                                                 :, commons.no_response_idx]
        mean_no_correct_response = main_avrge_behaviours[i,
                                                         :, commons.no_correct_response_idx]
        mean_correct_response = main_avrge_behaviours[i,
                                                      :, commons.correct_response_idx]
        plot_behs_graph(mean_no_response, mean_no_correct_response,
                        mean_correct_response, dataset, es, xtags=commons.memory_fills, prefix='hetero_remember-')
        for j, f in enumerate(commons.memory_fills):
            save_conf_matrix(
                main_avrge_confrixes[i, j], dataset, f'hetero_remember-fll_{str(f).zfill(3)}', es)
    print('Remembering done!')


def check_consistency(filling, es):
    print('Validating consistency using ' + 
          ('filling' if filling else 'testing') + ' data...' )
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    # We are capturing left and right measures.
    total_entropies = np.zeros((testing_folds, 2, len(memory_fills)))
    total_precisions = np.zeros((testing_folds, 2, len(memory_fills)))
    total_recalls = np.zeros((testing_folds, 2, len(memory_fills)))
    total_accuracies = np.zeros((testing_folds, 2, len(memory_fills)))
    total_confrixes = []
    total_behaviours = []

    for fold in range(testing_folds):
        fold, entropies, precisions, recalls, confrixes, behaviours = \
            check_consistency_per_fold(filling, es, fold)
        total_precisions[fold] = precisions
        total_recalls[fold] = recalls
        total_entropies[fold] = entropies
        total_confrixes.append(confrixes)
        total_behaviours.append(behaviours)
    total_confrixes = np.array(total_confrixes, dtype=int)
    total_behaviours = np.array(total_behaviours, dtype=int)

    main_avrge_entropies = np.mean(total_entropies, axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_precisions = np.mean(total_precisions, axis=0)
    main_stdev_precisions = np.std(total_precisions, axis=0)
    main_avrge_recalls = np.mean(total_recalls, axis=0)
    main_stdev_recalls = np.std(total_recalls, axis=0)
    main_avrge_accuracies = np.mean(total_accuracies, axis=0)
    main_stdev_accuracies = np.std(total_accuracies, axis=0)
    main_avrge_confrixes = np.mean(total_confrixes, axis=0)
    main_stdev_confrixes = np.std(total_confrixes, axis=0)
    main_avrge_behaviours = np.mean(total_behaviours, axis=0)
    main_stdev_behaviours = np.std(total_behaviours, axis=0)

    suffix = '-filling' if filling else '-testing'
    np.savetxt(
        commons.csv_filename(
            'validate_average_precision' + suffix, es),
        main_avrge_precisions, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'validate_average_recall' + suffix, es),
        main_avrge_recalls, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'validate_average_accuracy' + suffix, es),
        main_avrge_accuracies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'validate_average_entropy' + suffix, es),
        main_avrge_entropies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'validate_stdev_precision' + suffix, es),
        main_stdev_precisions, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'validate_stdev_recall' + suffix, es),
        main_stdev_recalls, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'validate_stdev_accuracy' + suffix, es),
        main_stdev_accuracies, delimiter=',')
    np.savetxt(
        commons.csv_filename(
            'validate_stdev_entropy' + suffix, es),
        main_stdev_entropies, delimiter=',')
    np.save(commons.data_filename('validate_mean_behaviours' + suffix, es),
            main_avrge_behaviours)
    np.save(commons.data_filename('validate_stdv_behaviours' + suffix, es),
            main_stdev_behaviours)
    np.save(commons.data_filename('validate_mean_confrixes' + suffix, es),
            main_avrge_confrixes)
    np.save(commons.data_filename('validate_stdv_confrixes-' + suffix, es),
            main_stdev_confrixes)

    for i in range(len(commons.datasets)):
        dataset = commons.datasets[i]
        plot_prerec_graph(
            100*main_avrge_precisions[i], 100 *
            main_avrge_recalls[i], main_avrge_entropies[i],
            100*main_stdev_precisions[i], 100*main_stdev_recalls[i], dataset,
            es, acc_mean=100*main_avrge_accuracies[i], acc_std=100*main_stdev_accuracies[i],
            prefix='hetero_validate' + suffix, xlabels=commons.memory_fills,
            xtitle=_('Percentage of memory corpus'))
        mean_no_response = main_avrge_behaviours[i,
                                                 :, commons.no_response_idx]
        mean_no_correct_response = main_avrge_behaviours[i,
                                                         :, commons.no_correct_response_idx]
        mean_correct_response = main_avrge_behaviours[i,
                                                      :, commons.correct_response_idx]
        plot_behs_graph(mean_no_response, mean_no_correct_response,
                        mean_correct_response, dataset, es, xtags=commons.memory_fills,
                        prefix='hetero_validate' + suffix)
        for j, f in enumerate(commons.memory_fills):
            save_conf_matrix(
                main_avrge_confrixes[i, j], dataset,
                f'hetero_validate-fll_{str(f).zfill(3)}' + suffix, es)
    print('Validation done!')

def decode_test_features(es):
    """ Creates images directly from test features.

    Uses the decoder part of the neural networks to (re)create
    images from features generated by the encoder.
    """
    for dataset in commons.datasets:
        model_prefix = commons.model_name(dataset, es)
        suffix = commons.testing_suffix
        testing_features_prefix = commons.features_name(dataset, es) + suffix
        suffix = commons.noised_suffix
        noised_features_prefix = commons.features_name(dataset, es) + suffix

        for fold in range(commons.n_folds):
            # Load test features and labels
            testing_features_filename = commons.data_filename(
                testing_features_prefix, es, fold)
            testing_features = np.load(testing_features_filename)
            testing_data, testing_labels = ds.get_testing(dataset, fold)
            noised_features_filename = commons.data_filename(
                noised_features_prefix, es, fold)
            noised_features = np.load(noised_features_filename)
            noised_data, _ = ds.get_testing(dataset, fold, noised=True)

            # Loads the decoder.
            model_filename = commons.decoder_filename(model_prefix, es, fold)
            model = tf.keras.models.load_model(model_filename)
            model.summary()
            # Generate images.
            prod_test_images = model.predict(testing_features)
            prod_nsed_images = model.predict(noised_features)
            n = len(testing_labels)
            # Save images.
            for (i, testing, prod_test, noised, prod_noise, label) in \
                    zip(range(n), testing_data, prod_test_images, noised_data,
                        prod_nsed_images, testing_labels):
                store_original_and_test(
                    testing, prod_test, noised, prod_noise,
                    commons.testing_path, i, label, dataset, es, fold)


def decode_memories(msize, es):

    for dataset in commons.datasets:
        model_prefix = commons.model_name(dataset, es)
        testing_labels_prefix = commons.labels_prefix + commons.testing_suffix
        memories_prefix = commons.memories_name(es)
        noised_prefix = commons.noised_memories_name(es)
        for fold in range(commons.n_folds):
            # Load test features and labels
            memories_features_filename = commons.data_filename(
                memories_prefix, es, fold)
            noised_features_filename = commons.data_filename(
                noised_prefix, es, fold)
            testing_labels_filename = commons.data_filename(
                testing_labels_prefix, es, fold)
            memories_features = np.load(memories_features_filename)
            noised_features = np.load(noised_features_filename)
            testing_labels = np.load(testing_labels_filename)
            # Loads the decoder.
            model_filename = commons.decoder_filename(model_prefix, es, fold)
            model = tf.keras.models.load_model(model_filename)
            model.summary()

            memories_images = model.predict(memories_features)
            noised_images = model.predict(noised_features)
            n = len(testing_labels)
            memories_path = commons.memories_path
            for (i, memory, noised, label) in \
                    zip(range(n), memories_images, noised_images, testing_labels):
                store_memory(memory, memories_path, i, label, es, fold)
                store_noised_memory(noised, memories_path, i, label, es, fold)


def store_original_and_test(testing, prod_test, noised, prod_noise,
                            test_dir, idx, label, dataset, es, fold):
    directory = os.path.join(test_dir, dataset)
    testing_filename = commons.testing_image_filename(
        directory, idx, label, es, fold)
    prod_test_filename = commons.prod_testing_image_filename(
        directory, idx, label, es, fold)
    noised_filename = commons.noised_image_filename(
        directory, idx, label, es, fold)
    prod_noise_filename = commons.prod_noised_image_filename(
        directory, idx, label, es, fold)
    store_image(testing_filename, testing)
    store_image(prod_test_filename, prod_test)
    store_image(noised_filename, noised)
    store_image(prod_noise_filename, prod_noise)


def store_memory(memory, directory, idx, label, es, fold):
    filename = commons.memory_image_filename(directory, idx, label, es, fold)
    full_directory = commons.dirname(filename)
    commons.create_directory(full_directory)
    store_image(filename, memory)


def store_noised_memory(memory, directory, idx, label, es, fold):
    memory_filename = commons.noised_image_filename(
        directory, idx, label, es, fold)
    store_image(memory_filename, memory)


def store_image(filename, array):
    pixels = array.reshape(ds.columns, ds.rows)
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(filename)


def features_parameters(suffix, dataset, es):
    cols = commons.datasets_to_domains[dataset]
    means = np.zeros((commons.n_folds, commons.n_labels, cols))
    stdvs = np.zeros((commons.n_folds, commons.n_labels, cols))
    for fold in range(commons.n_folds):
        features_filename = commons.features_name(dataset, es) + suffix
        features_filename = commons.data_filename(features_filename, es, fold)
        labels_filename = commons.labels_name(dataset, es) + suffix
        labels_filename = commons.data_filename(labels_filename, es, fold)
        labels = np.load(labels_filename)
        features = np.load(features_filename)
        for label in commons.all_labels:
            feats = np.array(
                [f for f, l in zip(features, labels) if l == label]
            )
            mean = np.mean(feats, axis=0)
            stdv = np.std(feats, axis=0)
            means[fold, label] = mean
            stdvs[fold, label] = stdv
        print(f'Means: {means[fold]}')
    return means, stdvs


def save_prototypes(means, suffix, dataset, es):
    proto_suffix = suffix + commons.proto_suffix
    for fold in range(commons.n_folds):
        proto_filename = commons.features_name(dataset, es) + proto_suffix
        proto_filename = commons.data_filename(proto_filename, es, fold)
        np.save(proto_filename, means[fold])
        # Loads the decoder.
        model_prefix = commons.model_name(dataset, es)
        model_filename = commons.decoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(model_filename)
        model.summary()
        proto_images = model.predict(means[fold])
        prototypes_path = commons.prototypes_path + commons.dataset_suffix(dataset) + suffix
        for (memory, label) in zip(proto_images, commons.all_labels):
            store_memory(memory, prototypes_path, label, label, es, fold)


def save_features_graphs(means, stdevs, dataset, es):
    for fold in range(commons.n_folds):
        plot_features_graph(
            commons.datasets_to_domains[dataset],
            means[fold],stdevs[fold], dataset, es)
        

##############################################################################
# Main section

def create_and_train_network(dataset, es):
    print(f'Memory size (columns): {commons.domain(dataset)}')
    model_prefix = commons.model_name(dataset, es)
    stats_prefix = commons.stats_model_name(dataset, es)
    history, conf_matrix = neural_net.train_network(dataset, model_prefix, es)
    save_history(history, stats_prefix, es)
    save_conf_matrix(conf_matrix, '',  stats_prefix, es, vmax=1.0)

def produce_features_from_data(dataset, es):
    model_prefix = commons.model_name(dataset, es)
    features_prefix = commons.features_name(dataset, es)
    labels_prefix = commons.labels_name(dataset, es)
    data_prefix = commons.data_name(dataset, es)
    neural_net.obtain_features(dataset,
                               model_prefix, features_prefix, labels_prefix, data_prefix, es)

def characterize_features(dataset, es):
    """ Produces a graph of features averages and standard deviations.
    """
    means, stdevs = features_parameters(commons.filling_suffix, dataset, es)
    save_prototypes(means, commons.filling_suffix, dataset, es)
    save_features_graphs(means, stdevs, dataset, es)

    
def describe_dataset(dataset, es):
    statistics(dataset, es)
    distances(dataset, es)

def run_separate_evaluation(dataset, es):
    best_memory_sizes = test_memory_sizes(dataset, es)
    print(f'Best memory sizes: {best_memory_sizes}')
    best_filling_percents = test_memory_fills(
        best_memory_sizes, dataset, es)
    save_learned_params(best_memory_sizes, best_filling_percents, dataset, es)

def run_evaluation(es):
    test_hetero_fills(es)

def generate_memories(es):
    # decode_test_features(es)
    remember(es)
    # decode_memories(es)

def validate(filling, es):
    check_consistency(filling, es)

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['--relsmean']:
        n = int(args['--relsmean'])
        if n <= 0:
            print('The mean of relations must be a positive integer.')
            exit(1)
        else:
            commons.mean_matches = n
    if args['--relsstdv']:
        s = float(args['--relsstdv'])
        if s <= 0:
            print('The standard deviation of relations must be a positive number.')
            exit(1)
        else:
            commons.stdv_matches = n
    if args['--runpath']:
        commons.run_path = args['--runpath']
    if args['es']:
        es_lang = gettext.translation(
            'eam', localedir='locale', languages=['es'])
        es_lang.install()

    if args['-p']:
        commons.use_prototypes = True
        
    # Reading memories parameters
    _prefix = commons.memory_parameters_prefix
    _filename = commons.csv_filename(_prefix)
    parameters = None
    try:
        parameters = \
            np.genfromtxt(_filename, dtype=float, delimiter=',', skip_header=1)
    except:
        pass

    exp_settings = commons.ExperimentSettings(params = parameters)
    print(f'Working directory: {commons.run_path}')
    print(f'Experimental settings: {exp_settings}')

    # PROCESSING OF MAIN OPTIONS.
    random.seed(0)
    if args['-n']:
        _dataset = args['<dataset>']
        if _dataset in commons.datasets:
            create_and_train_network(_dataset, exp_settings)
        else:
            print(f'Dataset {_dataset} is not supported.')
    elif args['-f']:
        _dataset = args['<dataset>']
        if _dataset in commons.datasets:
            produce_features_from_data(_dataset, exp_settings)
        else:
            print(f'Dataset {_dataset} is not supported.')
    elif args['-c']:
        _dataset = args['<dataset>']
        if _dataset in commons.datasets:
            characterize_features(_dataset, exp_settings)
        else:
            print(f'Dataset {_dataset} is not supported.')
    elif args['-d']:
        _dataset = args['<dataset>']
        if _dataset in commons.datasets:
            describe_dataset(_dataset, exp_settings)
        else:
            print(f'Dataset {_dataset} is not supported.')
    elif args['-s']:
        _dataset = args['<dataset>']
        if _dataset in commons.datasets:
            run_separate_evaluation(_dataset, exp_settings)
        else:
            print(f'Dataset {_dataset} is not supported.')
    elif args['-e']:
        run_evaluation(exp_settings)
    elif args['-r']:
        generate_memories(exp_settings)
    elif args['-v']:
        validate(True, exp_settings)
    elif args['-w']:
        validate(False, exp_settings)

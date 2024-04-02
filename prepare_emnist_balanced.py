"""Preparation of EMNIST Balanced dataset.

Usage:
  eam -h | --help
  eam (-r | -s)

Options:
  -h    Show this screen.
"""
import os
import random
import numpy as np
import gzip
from docopt import docopt
import commons

def save_mnist_like(images, labels, dirname, subset):
    labels_path = os.path.join(dirname, f'{subset}-labels-idx1-ubyte')
    images_path = os.path.join(dirname, f'{subset}-images-idx3-ubyte')
    header = np.array([0x0801, len(labels)], dtype='>i4')
    with open(labels_path, "wb") as f:
        f.write(header.tobytes())
        f.write(labels.tobytes())
    header = np.array([0x0803, len(images), 28, 28], dtype='>i4')
    with open(images_path, "wb") as f:
        f.write(header.tobytes())
        f.write(images.tobytes())
    
def load_mnist_like(dirname, subset, transposed):
    labels_path = os.path.join(dirname, f'{subset}-labels-idx1-ubyte.gz')
    images_path = os.path.join(dirname, f'{subset}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),
            dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),
            dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
    if transposed:
        images = transpose(images)
    return images, labels

def transpose(images):
    transposed = []
    for image in images:
        transposed.append(np.transpose(image))
    return np.array(transposed)

def filter_and_remap(images, labels, map):
    chosen_images = []
    chosen_labels = []
    for image, label in zip(images, labels):
        if label in map.keys():
            chosen_images.append(image)
            chosen_labels.append(map[label])
    return np.array(chosen_images, dtype=np.uint8), np.array(chosen_labels, np.uint8)
    

if __name__=='__main__':
    random.seed(0)
    args = docopt(__doc__)
    dirname = os.path.join(commons.data_path, 'emnist')
    if args['-r']:
        orig_prefix = 'emnist-balanced-'
        dest_prefix = 'emnist-uppercase-'
        chosen = list(range(10,36))
        transposed = True
    elif args['-s']:
        orig_prefix = 'emnist-uppercase-'
        dest_prefix = ''
        chosen = [19, 14, 15, 1, 22, 25, 12, 0, 11, 18]
        transposed = False
    else:
        exit(1)
    for subset in ['test', 'train']:
        name = orig_prefix + subset
        images, labels = load_mnist_like(dirname, name, transposed)
        unique, counts = np.unique(labels, return_counts=True)
        print(f'Original labels: {unique}')
        print(f'Original frequencies in {name}: {counts}')
        print(f'Chosen labels: {chosen}')
        map = {}
        for i in range(len(chosen)):
            map[chosen[i]] = i
        images, labels = filter_and_remap(images, labels, map)
        unique, counts = np.unique(labels, return_counts=True)
        name = dest_prefix + subset
        print(f'New labels in {name}: {unique}')
        print(f'New frequencies in {name}: {counts}')
        save_mnist_like(images, labels, dirname, name)


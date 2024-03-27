import os
import random
import numpy as np
import gzip
import commons


def save_mnist_like(images, labels, dirname, kind):
    labels_path = os.path.join(dirname, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(dirname, '%s-images-idx3-ubyte' % kind)
    header = np.array([0x0801, len(labels)], dtype='>i4')
    with open(labels_path, "wb") as f:
        f.write(header.tobytes())
        f.write(labels.tobytes())
    header = np.array([0x0803, len(images), 28, 28], dtype='>i4')
    with open(images_path, "wb") as f:
        f.write(header.tobytes())
        f.write(images.tobytes())
    
def load_mnist_like(dirname, kind):
    labels_path = os.path.join(dirname, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(dirname, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),
            dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),
            dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
    return images, labels

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
    dirname = os.path.join(commons.data_path, 'emnist')
    images, labels = load_mnist_like(dirname, 'train')
    unique, counts = np.unique(labels, return_counts=True)
    print(f'Original labels: {unique}')
    print(f'Original frequencies in train: {counts}')
    chosen = np.array([15, 13, 23, 16, 26, 19, 3, 20, 2, 24]) # np.random.choice(unique, commons.n_labels, False)
    print(f'Chosen labels: {chosen}')
    map = {}
    for i in range(commons.n_labels):
        map[chosen[i]] = i
    images, labels = filter_and_remap(images, labels, map)
    unique, counts = np.unique(labels, return_counts=True)
    print(f'New labels: {unique}')
    print(f'New frequencies in train: {counts}')
    save_mnist_like(images, labels, dirname, 'newtrain')
    images, labels = load_mnist_like(dirname, 'test')
    unique, counts = np.unique(labels, return_counts=True)
    print(f'Original frequencies in test: {counts}')
    images, labels = filter_and_remap(images, labels, map)
    unique, counts = np.unique(labels, return_counts=True)
    print(f'New frequencies in test: {counts}')
    save_mnist_like(images, labels, dirname, 't10k')


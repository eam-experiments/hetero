import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import commons

runpath = 'runs_4d'
filename = 'presence'

def plot_presence(xtags, means, stdev, skewn, kurto, dataset, fname):
    ymax = 100.0
    x = np.arange(0.0, ymax, ymax/len(xtags))
    width = ymax / (1.5*means.size)
    # It is assumed means.shape[1] = 2
    plt.bar(x-width/2, means[:, 0]*ymax, width=width, label='Expected')
    plt.errorbar(x-width/2, means[:, 0]*ymax, stdev[:, 0]*ymax, fmt='o', color='k')
    plt.bar(x+width/2, means[:, 1]*ymax, width=width, label='Best Other')
    plt.errorbar(x+width/2, means[:, 1]*ymax, stdev[:, 1]*ymax, fmt='o', color='k')
    plt.ylim(0.0, ymax)
    plt.xticks(x, xtags)
    plt.xlabel('Filling percent')
    plt.ylabel('Presence')
    plt.title(dataset)
    plt.legend(loc='best')
    plt.grid(axis='y')

    fname = fname + '.svg'
    plt.savefig(fname, dpi=600)
    plt.close()

def only_general(stats):
    general = []
    i = 0
    while i < stats.shape[0]:
        general.append(stats[i])
        i += 3
    return np.array(general)

def gen_graph(stats, dataset, fname):
    general = only_general(stats)
    fillings = general[:, 0]
    means = general[:, 1:3]    
    stdev = general[:, 3:5]    
    skewn = general[:, 5:7]    
    kurto = general[:, 7:9]    
    plot_presence(fillings, means, stdev, skewn, kurto, dataset, fname)

if __name__ == "__main__":
    for dataset in commons.datasets:
        fname = runpath + '/' + filename + '-' + dataset
        stats = np.genfromtxt(fname + '.csv', dtype=float, delimiter=',', skip_header=1)
        print(f'Shape of input data: {stats.shape}')
        gen_graph(stats, dataset, fname)
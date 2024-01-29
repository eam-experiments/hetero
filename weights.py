# Copyright [2023] Rafael Morales Gamboa, and
# Luis Alberto Pineda Cortés.
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

"""Generates a graph comparing weights of correctly and incorrectly classified output

Usage:
    weights -h | --help
    weights <csv_fname> [--runpath=PATH]

Options:
    -h --help           Show this information.
    --runpath=PATH      Directory where to find the csv file and save the graphs.
"""
from docopt import docopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import commons

def plot_graph(cr_means, cr_stdvs, ir_means, ir_stdvs, name):        
    tags = commons.memory_fills
    x_pos = np.arange(len(tags))
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    cr = ax.bar(x_pos - width/2, cr_means, width, yerr=cr_stdvs, label='Correctly recognized')
    ir = ax.bar(x_pos + width/2, ir_means, width, yerr=ir_stdvs, label='Incorrectly recognized')
    ax.set_xlabel('Range Quantization Level')
    ax.set_ylabel('Average Weight')
    ax.set_title('Weight Comparison Between Groups')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tags)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='y')
    ax.legend()
    # autolabel(ax, cr, "left")
    # autolabel(ax, ir, "right")
    fig.tight_layout()
    fname = 'weight_comparison-' + name
    graph_filename = commons.picture_filename(fname)
    plt.savefig(graph_filename, dpi=600)
    plt.show()

def autolabel(ax, bars, pos='center'):
    """
    Attach a text label above each bar in `bars`, displaying its height.

    `pos` indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(offset[pos]*3, 3),  # use 3 points offset
                    textcoords='offset points',  # in both directions
                    ha=ha[pos], va='bottom')

def describe_weights(df):
    mnist = df.iloc[::2]  # even
    fashion = df.iloc[1::2]  # odd
    plot_graph(mnist['CorClasMean'], mnist['CorClasStdv'],
               mnist['IncClasMean'], mnist['IncClasStdv'], 'mnist2fashion')
    plot_graph(fashion['CorClasMean'], fashion['CorClasStdv'],
               fashion['IncClasMean'], fashion['IncClasStdv'], 'fashion2mnist')

if __name__ == "__main__":
    args = docopt(__doc__)
    fname = args['<csv_fname>']
    if args['--runpath']:
        commons.run_path = args['--runpath']
    csv_fname = commons.csv_filename(fname)
    df = pd.read_csv(csv_fname)
    describe_weights(df)
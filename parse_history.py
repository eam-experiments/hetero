# Copyright [2024] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

"""Parsing of neural networks performance history

Usage:
  parse_history -h | --help
  parse_history <filename>

Options:
  -h    Show this screen.
  <fname> Name of the file with the JSON data.
"""
from docopt import docopt
import json

class_metric = 'accuracy'
autor_metric = 'decoder_root_mean_squared_error'

if __name__ == "__main__":
    args = docopt(__doc__)
    filename = args['<filename>']
    with open(filename, 'r') as f:
        data = json.load(f)
        history = data['history']
        # In every three, the first element is the trace of the training, 
        # and it is ignored. The second and third elements contain
        # the metric and loss for the classifier and autoencoder,
        # respectively
        print(f'History lenght: {len(history)}')
        for i in range(0, len(history), 3):
            fold = int(i/3)
            class_value = history[i+1][class_metric]
            autor_value = history[i+2][autor_metric]
            print(f'{fold},{class_value},{autor_value}')
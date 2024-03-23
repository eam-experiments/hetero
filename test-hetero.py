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

import random
import numpy as np
import commons
from hetero_associative_4d import HeteroAssociativeMemory4D

es = commons.ExperimentSettings()
h = HeteroAssociativeMemory4D(2,3,3,2, es)
random.seed()

# Memory for associations between functions of two characteristics and three values,
# to functions of three characteristics and two values
print("Original state:")
print(h)

v0 = np.array([0, 0])
v1 = np.array([1, 1])
v2 = np.array([2, 2])

w0 = np.array([0, 0, 0])
w1 = np.array([1, 1, 1])

# 0: 1 0
# 1: 0 1
# 2: 0 0
vd = np.array([0, 1])

# 0: 1 0 1
# 1: 0 1 0
wd = np.array([0, 1, 0])

# 0: 0 0
# 1: 0 1
# 2: 1 0
vi = np.array([2, 1])

# 0: 0 1 0
# 1: 1 0 1
wi = np.array([1, 0, 1])

vu = np.array([1, 100])
wn = np.array([1, np.nan, 1])

print(f'v0: {v0}')
print(f'w0: {w0}')
print("h.register(v0,w0)")
h.register(v0,w0)
print(h)

print("h.register(v0,w0)")
h.register(v0,w0)
print(h)

print(f'v1: {v1}')
print(f'w1: {w1}')
print("h.register(v1,w1)")
h.register(v1,w1)
print(h)

# print(f'v2: {v2}')
# print(f'w0: {w0}')
# print("h.register(v2,w0)")
# h.register(v2,w0)
# print(h)

print(f'vd: {vd}')
print(f'w0: {w0}')

print("h.register(vd,w0)")
h.register(vd, w0)
print(h)

print('h.recognize(v0,w0):')
r = h.recognize(v0,w0)
print(f'Result: {r}')

print('h.recognize(v1,w1):')
r = h.recognize(v1,w1)
print(f'Result: {r}')

print('h.recognize(v0,w1):')
r = h.recognize(v0,w1)
print(f'Result: {r}')

print('h.recognize(v1,w0):')
r = h.recognize(v1,w0)
print(f'Result: {r}')

if h.model_name == commons.d4_model_name:
    print(f'vu: {vu}')
    print(f'wn: {wn}')

    print('h.recognize(vu,wn):')
    r = h.recognize(vu,wn)
    print(f'Result: {r}')

print(f'v0: {v0}')
print('r = h.recall_from_left(v0)')
r = h.recall_from_left(v0)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'v1: {v1}')
print('r = h.recall_from_left(v1)')
r = h.recall_from_left(v1)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'w0: {w0}')
print('r = h.recall_from_right(w0)')
r = h.recall_from_right(w0)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'vd: {vd}')
print('r = h.recall_from_left(vd)')
r = h.recall_from_left(vd)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'w1: {w1}')
print('r = h.recall_from_right(w1)')
r = h.recall_from_right(w1)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'wi: {wi}')
print('r = h.recall_from_right(wi)')
r = h.recall_from_right(wi)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'v2: {v2}')
print('r = h.recall_from_left(v2)')
r = h.recall_from_left(v2)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'vu: {vu}')
print('r = h.recall_from_left(vu)')
r = h.recall_from_left(vu)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

print(f'wn: {wn}')
print('r = h.recall_from_right(wn)')
r = h.recall_from_right(wn)
print(f'vector: {r[0] if r[1] else None}, recognized: {r[1]}, weight: {r[2]}')
print('Projection:')
if h.model_name == commons.d4_model_name:
    commons.print_csv(r[3])
else:
    print(h.relation_to_string(r[3]))

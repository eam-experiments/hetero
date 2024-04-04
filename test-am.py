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

import numpy as np
from associative import AssociativeMemory

# Memory for 4 features and 5 values
m = AssociativeMemory(4,5)

v0 = np.full(4, 0, dtype=int)
v1 = np.full(4, 1, dtype=int)
v2 = np.full(4, 2, dtype=int)
v3 = np.full(4, 3, dtype=int)
v4 = np.full(4, 4, dtype=int)

vd = np.array([0, 1, 2, 3])
vi = np.array([3, 2, 1, 0])
vu = np.array([10,-1,0, 1])

m.register(v0)
print('m.register(v0):')
print(m)
m.register(v1)
print('m.register(v1):')
print(m)
m.register(v4)
print('m.register(v4):')
print(m)
for _ in range(10):
    m.register(vd)
    m.register(vi)
print('m.register(vd and vi) ten times:')
print(m)

print('m.recognize(v0):')
print(m.recognize(v0))
print('m.recognize(v1):')
print(m.recognize(v1))
print('m.recognize(v2):')
print(m.recognize(v2))
print('m.recognize(v3):')
print(m.recognize(v3))
print('m.recognize(v4):')
print(m.recognize(v4))
print('m.recognize(vd):')
print(m.recognize(vd))
print('m.recognize(vi):')
print(m.recognize(vi))
print('m.recognize(vu):')
print(m.recognize(vu))

print('m.recall(v0):')
print(m.recall(v0))
print('m.recall(v1):')
print(m.recall(v1))
print('m.recall(v2):')
print(m.recall(v2))
print('m.recall(v3):')
print(m.recall(v3))
print('m.recall(v4):')
print(m.recall(v4))
print('m.recall(vd):')
print(m.recall(vd))
print('m.recall(vi):')
print(m.recall(vi))
print('m.recall(vu):')
print(m.recall(vu))

print('m.recall():')
print(m.recall())
print('m.recall():')
print(m.recall())
print('m.recall():')
print(m.recall())






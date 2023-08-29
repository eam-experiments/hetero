# Copyright (C) 2023 Rafael Morales-Gamboa
# 
# This file is part of hetero.
# 
# hetero is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# hetero is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with hetero.  If not, see <http://www.gnu.org/licenses/>.

# Code taken from https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice

import random

class CustomSet:
    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose(self):
        return random.choice(self.items)
    
if __name__ == "__main__":
    s = CustomSet()
    s.add(1)
    s.add(2)
    s.remove(1)
    print(s.choose())
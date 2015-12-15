__author__ = 'harshad'

import numpy as np

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.own_matrix = np.ndarray([])
        self.column_number = 0
        self.val = 0
        self.left_matrix = np.ndarray([])
        self.children_map =  { }
        self.right_matrix = np.ndarray([])
        self.is_leaf = False
        self.decision_value = 0

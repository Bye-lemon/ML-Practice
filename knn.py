#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from utils.distance import *


class KDNode(object):
    def __init__(self, data: np.array = None):
        self.data = data
        self.left = None
        self.right = None

    def __repr__(self):
        return f"<KD-Node {self.data}>"


class KDTree(object):
    """
    Blanced kd-tree
    """

    def __init__(self, data: np.array) -> np.array:
        self.tree = KDNode(data)
        self.dim = data.shape[1]
        self.point = -1
        self.build(self.tree)

    def build(self, node: KDNode):
        self.point += 1
        if node.data is None:
            return
        if len(node.data) == 1:
            node.left = None
            node.right = None
        else:
            feature = self.point % self.dim
            data_st = sorted(node.data, key=lambda x: x[feature])
            left = data_st[: len(data_st) // 2]
            mid = data_st[len(data_st) // 2]
            right = data_st[len(data_st) // 2 + 1:]
            node.left = KDNode(left) if len(left) > 0 else KDNode()
            node.data = [np.array(mid)]
            node.right = KDNode(right) if len(right) > 0 else KDNode()
            self.build(node.left)
            self.point -= 1
            self.build(node.right)
            self.point -= 1

    def _travel(self, node: KDNode):
        if node.left is not None:
            self._travel(node.left)
        print(node.data)
        if node.right is not None:
            self._travel(node.right)

    def travel(self):
        self._travel(self.tree)


class KNNClassifier(object):
    """
    K-nearest Neighbor Algorithm
    """

    def __init__(self, data: np.array):
        super().__init__()
        self.kdtree = KDTree(data=data)
        self.dim = self.kdtree.dim

    def _predict(self, x: np.array, node: KDNode, choice: KDNode, depth: int = 0) -> np.array:
        if choice.data is None or lpdistance(x, node.data[0], p=2) < lpdistance(choice.data[0], node.data[0], p=2):
            choice = node
        if x[depth % self.dim] < node.data[0][depth % self.dim]:
            if node.left is not None:
                return self._predict(x, node.left, choice, depth + 1)
            else:
                return node
        else:
            if node.right is not None:
                return self._predict(x, node.right, choice, depth + 1)
            else:
                return node

    def predict(self, x: np.array) -> np.array:
        return np.array(list(map(lambda sample: self._predict(sample, self.kdtree.tree, choice=KDNode(), depth=0), x)))


if __name__ == "__main__":
    t = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    knn = KNNClassifier(t)
    res = knn.predict(np.array([[1, 1], [4, 8]]))
    print(res)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class NaiveBayesClassifier(object):
    """
    Naive Bayes Algorithm
    """
    def __init__(self, data: np.array, label: np.array):
        super().__init__()
        self.data = data
        self.label = label
        self.py = None
        self.pxy = None
        self._build()

    def _build(self):
        self.y_set = set(y for y in self.label)
        self.py = {y: np.sum(self.label == y) / len(self.label) for y in self.y_set}
        self.pxy = dict({})
        self.x_set = list(map(set, self.data.T))
        for y in self.y_set:
            mask = list()
            for i, x in enumerate(self.data):
                if self.label[i] == y:
                    mask.append(x)
            foo = np.array(mask).T
            apx = dict({})
            for i, x in enumerate(self.x_set):
                px = dict({_: np.sum(foo[i] == _) / len(foo[i]) for _ in x})
                apx.update({i: px})
            self.pxy.update({y: apx})

    def _predict(self, sample):
        res = dict({})
        for y in self.y_set:
            p = self.py[y]
            for i, x in enumerate(sample):
                p *= self.pxy[y][i][x]
            res.update({p: y})
        return res[max(res.keys())]

    def predict(self, samples: np.array) -> np.array:
        return np.array(list(map(self._predict, samples)))

    def show(self):
        print(self.py)
        print(self.pxy)


if __name__ == "__main__":
    data = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
                     [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
                     [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    label = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    bayes = NaiveBayesClassifier(data, label)
    bayes.show()
    res = bayes.predict(np.array([[2, 'S']]))
    print(res)

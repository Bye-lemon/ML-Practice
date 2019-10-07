#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class Preceptron(object):
    """
    One Layer Preceptron
    """

    def __init__(self):
        self.w = None
        self.b = 0

    def __repr__(self):
        return f"<OneLayerPreceptron y = {self.w}x + {self.b}>"

    def train(self, x: np.array, y: np.array, lr: float) -> None:
        self.w = np.zeros(x.shape[1])
        epoch = 1
        wrong = 1
        while wrong:
            wrong = 0
            for index, record in enumerate(x):
                if (np.dot(self.w, record.T) + self.b) * y[index] <= 0:
                    self.w += lr * y[index] * record
                    self.b += lr * y[index]
                    wrong += 1
            print(f"[Training] Epoch {epoch}: Wrong Point - {wrong} w - {self.w} b - {self.b}")
            epoch += 1
        print(f"[Training] Finish w - {self.w} b - {self.b}")

    def predict(self, x: np.array) -> np.array:
        _predict = lambda x: 1 if np.dot(self.w, x.T) + self.b > 0 else -1
        res = map(_predict, x)
        return np.array(list(res))


if __name__ == "__main__":
    x = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    model = Preceptron()
    model.train(x, y, lr=1)
    res = model.predict(np.array([[0, 0], [5, 5]]))
    print(model)
    print(res)

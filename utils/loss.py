#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def ZeroOneLoss(y: np.array, y_hat: np.array) -> int:
    """
    0-1 Loss Function
    """

    return np.sum((y - y_hat) != 0)


def QuadraticLoss(y: np.array, y_hat: np.array) -> float:
    """
    Quadratic Loss Function
    """
    return np.sum((y - y_hat) ** 2)


def AbsoluteLoss(y: np.array, y_hat: np.array) -> float:
    """
    Absolute Loss Fuction
    """
    return np.sum(abs(y - y_hat))

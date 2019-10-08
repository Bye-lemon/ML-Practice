#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def lpdistance(x: np.array, y: np.array, p: int) -> np.array:
    return np.sum((x - y) ** p) ** (1 / p)

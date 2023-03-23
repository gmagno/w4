#!/usr/bin/env python

"""Tests for `Rosenbrock`'s problem."""

import numpy as np
import pytest

from w4.w4 import solution_dtype, w4
from w4.xy import Decomposition


def f(x: np.ndarray) -> np.ndarray:
    return np.array((10 * (x[1] - x[0] ** 2), 1 - x[0]))


def fa(x: np.ndarray) -> np.ndarray:
    return np.array((abs(10 * x[1]) + abs(-10 * x[0] ** 2), abs(1) + abs(-x[0])))


def jac(x: np.ndarray) -> np.ndarray:
    return np.array([[-20 * x[0] - 1, 10], [-1, 0]])


@pytest.mark.parametrize(
    "w4_kwargs,expected_output",
    [
        (
            dict(x0=np.array((1.2, 1.0)), decomposition=Decomposition.SV),
            np.array(
                [(9, 6.747541260676108e-05, 1.00002179, 0.99990862)],
                dtype=solution_dtype(dim=2),
            ),
        ),
    ],
)
def test_rosenbrock(w4_kwargs: dict, expected_output: np.ndarray) -> None:
    solution: np.ndarray = w4(
        f=f,
        fa=fa,
        jac=jac,
        dt=0.8,
        maxiter=pow(10, 7),
        errmax=pow(10, -4),
        trace=False,
        **w4_kwargs
    )
    assert all(
        [
            np.allclose(solution[name], expected_output[name])
            for name in solution.dtype.names
        ]
    )

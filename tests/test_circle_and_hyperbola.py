#!/usr/bin/env python

"""Tests for `Circle and Hyperbola` problem."""

import numpy as np
import pytest

from w4.w4 import solution_dtype, w4
from w4.xy import Decomposition


def f(x: np.ndarray) -> np.ndarray:
    return np.array((x[0] ** 2 + x[1] ** 2 - 4, x[0] ** 2 * x[1] - 1))


def fa(x: np.ndarray) -> np.ndarray:
    return np.array(
        (abs(x[0] ** 2) + abs(x[1] ** 2) + abs(-4), abs(x[0] ** 2 * x[1]) + abs(-1))
    )


def jac(x: np.ndarray) -> np.ndarray:
    return np.array([[2 * x[0], 2 * x[1]], [2 * x[0] * x[1], x[0] ** 2]])


@pytest.mark.parametrize(
    "w4_kwargs,expected_output",
    [
        (
            dict(x0=np.array((-5.0, 0.5)), decomposition=Decomposition.LU),
            np.array(
                [(23, 7.64550448e-05, -1.98381116, 0.25411751)],
                dtype=solution_dtype(dim=2),
            ),
        ),
        (
            dict(x0=np.array((-0.5, 5.0)), decomposition=Decomposition.LU),
            np.array(
                [(22, 6.12220725e-05, -0.73303954, 1.86087154)],
                dtype=solution_dtype(dim=2),
            ),
        ),
        (
            dict(x0=np.array((0.5, 5.0)), decomposition=Decomposition.LU),
            np.array(
                [(22, 6.12220725e-05, 0.73303954, 1.86087154)],
                dtype=solution_dtype(dim=2),
            ),
        ),
    ],
)
def test_circle_and_hiperbole(w4_kwargs: dict, expected_output: np.ndarray) -> None:
    solution: np.ndarray = w4(
        f=f, fa=fa, jac=jac, dt=0.5, maxiter=1000, errmax=1e-4, trace=False, **w4_kwargs
    )
    assert all(
        [
            np.allclose(solution[name], expected_output[name])
            for name in solution.dtype.names
        ]
    )

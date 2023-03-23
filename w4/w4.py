"""W4 Method for Nonlinear Root Finding"""

from typing import Callable

import numpy as np

from w4.xy import Decomposition, xy


def solution_dtype(dim: int) -> np.dtype:
    dtype: np.dtype = np.dtype(
        [
            ("iteration", np.uint64),
            ("error", np.float64),
            *((f"x{n}", np.float64) for n in range(dim)),
        ]
    )
    return dtype


def w4(
    x0: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    fa: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    dt: float = 0.5,
    maxiter: int = 1000,
    errmax: float = 1e-4,
    decomposition: Decomposition = Decomposition.LU,
    trace: bool = False,
) -> np.ndarray:
    x: np.ndarray = x0
    p: np.ndarray = np.zeros(x.shape)

    dtype: np.dtype = solution_dtype(dim=len(x))
    solution: np.ndarray = np.zeros(maxiter, dtype=dtype)

    for i in range(maxiter):

        jac_x: np.ndarray = jac(x)
        f_x: np.ndarray = f(x)
        fa_x: np.ndarray = fa(x)

        error: float = max(abs(f_x / fa_x))

        X: np.ndarray
        Y: np.ndarray
        X, Y = xy(jac=jac_x, decomposition=decomposition)

        x = x + dt * X @ p  # Eq. 30a
        p = (1 - 2 * dt) * p - dt * Y @ f_x  # Eq. 30b

        solution[i] = (i, error, *x)

        if error < errmax:
            break

    return solution[(0 if trace else i) : i + 1]


def main() -> None:
    x0: np.ndarray = np.array([0.5, 5.0])

    def f(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + x[1] ** 2 - 4, x[0] ** 2 * x[1] - 1])

    def fa(x: np.ndarray) -> np.ndarray:
        return np.array(
            [abs(x[0] ** 2) + abs(x[1] ** 2) + abs(-4), abs(x[0] ** 2 * x[1]) + abs(-1)]
        )

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0], 2 * x[1]], [2 * x[0] * x[1], x[0] ** 2]])

    solution: np.ndarray = w4(
        x0=x0, f=f, fa=fa, jac=jac, decomposition=Decomposition.LU, trace=True
    )
    print(solution)


if __name__ == "__main__":
    main()

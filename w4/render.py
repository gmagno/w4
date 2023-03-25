from typing import Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from w4.w4 import w4_matrix
from w4.xy import Decomposition


def plot_f(
    f: Callable[[np.ndarray], np.ndarray],
    axis_abs_range=5.0,
    roots: Optional[np.ndarray] = None,
) -> None:

    xlist: np.ndarray = np.linspace(-axis_abs_range, axis_abs_range, 100)
    ylist: np.ndarray = np.linspace(-axis_abs_range, axis_abs_range, 100)
    X: np.ndarray
    Y: np.ndarray
    X, Y = np.meshgrid(xlist, ylist)
    x: np.ndarray = np.array([X, Y])
    F: np.ndarray = f(x)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.contour(X, Y, F[0], [0], colors="C0", linestyles="solid")
    ax.contour(X, Y, F[1], [0], colors="C1", linestyles="solid")

    if roots is not None:
        ax.scatter(roots.T[0], roots.T[1], marker="x", c="k")

    ax.grid(True)


def plot_basin(
    solutions_matrix: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    maxiter: int,
    errmax: float,
) -> None:
    round_ndigits = int(np.log10(1 / errmax) - 1)
    roots: np.ndarray = np.array(
        sorted(
            {
                (round(s["x0"], round_ndigits), round(s["x1"], round_ndigits))
                for s in solutions_matrix.flatten()
                if not np.isnan(s["x0"])
                and not np.isnan(s["x1"])
                and s["iteration"] < maxiter - 1
            }
        )
    )

    img: np.ndarray = np.zeros((*solutions_matrix.shape, 3), dtype=np.float64)

    h: float
    s: float
    v: float
    k_min: int = min(solutions_matrix.flatten()["iteration"])
    # k_max: int = max(solutions_matrix.flatten()["iteration"])
    k_unique: np.ndarray = np.unique(solutions_matrix.flatten()["iteration"])
    k_max: int = k_unique[-1] if k_unique[-1] < maxiter - 1 else k_unique[-2]

    for i, j in np.ndindex(solutions_matrix.shape):
        k: int = solutions_matrix[i, j]["iteration"]
        if k >= maxiter - 1:
            h, s, v = 0.0, 0.0, 0.0
        elif np.isnan(tuple(solutions_matrix[i, j][["x0", "x1"]])).any():
            h, s, v = 0.0, 0.0, 1.0
        else:
            z: np.ndarray = np.array(tuple(solutions_matrix[i, j][["x0", "x1"]]))
            diff: np.ndarray = roots - z
            root_idx: int = int((abs(diff[:, 0]) + abs(diff[:, 1])).argmin())

            h = root_idx / len(roots)
            s = 1
            # v = 1 - k / maxiter
            v = 1 - (k - k_min) / k_max
        img[i, j] = (h, s, v)

    rgb = mpl.colors.hsv_to_rgb(img)
    nrows: int = solutions_matrix.shape[0]
    ncols: int = solutions_matrix.shape[1]
    rowsdiv: float = (ymax - ymin) / nrows
    colsdiv: float = (xmax - xmin) / ncols
    fig, ax = plt.subplots()
    ax.imshow(
        rgb,
        extent=[
            xmin - colsdiv / 2,
            xmax - colsdiv / 2,
            ymin - rowsdiv / 2,
            ymax - rowsdiv / 2,
        ],
    )
    ax.scatter(roots.T[0], roots.T[1], marker="x", c="k")


def main() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + x[1] ** 2 - 4, x[0] ** 2 * x[1] - 1])

    def fa(x: np.ndarray) -> np.ndarray:
        return np.array(
            [abs(x[0] ** 2) + abs(x[1] ** 2) + abs(-4), abs(x[0] ** 2 * x[1]) + abs(-1)]
        )

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0], 2 * x[1]], [2 * x[0] * x[1], x[0] ** 2]])

    width: int = 21
    height: int = 21
    xmin: float = -5.0
    xmax: float = 5.0
    ymin: float = -5.0
    ymax: float = 5.0
    maxiter: int = 1000
    errmax: float = 1e-4

    solutions_matrix: np.ndarray = w4_matrix(
        width=width,
        height=height,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        f=f,
        fa=fa,
        jac=jac,
        dt=0.5,
        maxiter=maxiter,
        errmax=errmax,
        decomposition=Decomposition.LU,
    )

    plot_basin(solutions_matrix, xmin, xmax, ymin, ymax, maxiter, errmax)

    plot_f(
        f,
        roots=np.array(
            [
                (-1.9838041, 0.25411772),
                (1.9838041, 0.25411772),
                (0.73310016, 1.86080149),
                (-0.73310016, 1.86080149),
            ]
        ),
    )
    plt.show()


if __name__ == "__main__":
    main()

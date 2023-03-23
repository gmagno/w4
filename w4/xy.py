from enum import Enum
from typing import Callable, NamedTuple

import numpy as np
from scipy import linalg


class Decomposition(str, Enum):
    LU = "lu"
    LH = "lh"
    SV = "sv"


class XY(NamedTuple):
    x: np.ndarray
    y: np.ndarray


def xy(jac: np.ndarray, decomposition: Decomposition = Decomposition.SV) -> XY:
    decomposition_to_func: dict[Decomposition, Callable[[np.ndarray], XY]] = {
        Decomposition.LU: xy_lu,
        Decomposition.LH: xy_lh,
        Decomposition.SV: xy_sv,
    }
    xy_func: Callable[[np.ndarray], XY] = decomposition_to_func[decomposition]
    return xy_func(jac)


def xy_lu(jac: np.ndarray) -> XY:
    jac_inv: np.ndarray = linalg.inv(jac)
    p: np.ndarray
    l: np.ndarray
    u: np.ndarray
    p, l, u = linalg.lu(jac_inv.T)
    x: np.ndarray = u.T
    y: np.ndarray = l.T @ p
    return XY(x, y)


def xy_lh(jac: np.ndarray) -> XY:
    """"""
    q: np.ndarray
    r: np.ndarray
    q, r = linalg.qr(jac.T)
    x: np.ndarray = q
    try:
        y: np.ndarray = linalg.inv(r.T)
    except linalg.LinAlgError:
        # r.T is singular
        ...  # TODO: handle exception
    return XY(x, y)


def xy_sv(jac: np.ndarray, tol=1e-6) -> XY:
    """"""
    u: np.ndarray
    s: np.ndarray
    vh: np.ndarray
    try:
        u, s, vh = linalg.svd(jac)
    except linalg.LinAlgError:
        # SVD computation did not converge
        # TODO: handle exception
        ...

    sis: np.ndarray = np.ones(len(s))
    sis[s > tol] = 1 / s[s > tol]
    s_diag: np.ndarray = np.diag(sis)
    x: np.ndarray = vh.T
    y: np.ndarray = s_diag @ u.T
    return XY(x, y)

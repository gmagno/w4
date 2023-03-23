W4
===

[![PyPI version shields.io](https://img.shields.io/pypi/v/w4.svg)](https://pypi.python.org/pypi/w4/)

[![PyPI license](https://img.shields.io/pypi/l/w4.svg)](https://pypi.python.org/pypi/w4/)

This package provides the [W4
method](https://doi.org/10.1016/j.apnum.2022.08.019) for nonlinear root finding, inspired by the [R implementation](https://github.com/ramiromagno/w4).

Install
-------

create a virtual environment, activate it and upgrade pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

install from pypi:

```bash
pip install w4
```

or from github repo:

```bash
pip install git+https://github.com/gmagno/w4
```

Usage
-----

```python
import numpy as np
from w4.w4 import w4
from w4.xy import Decomposition

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
```

Output:

```bash
[( 0, 7.26495726e-01, 0.5       , 5.        )
 ( 1, 7.26495726e-01, 0.51413317, 4.46733668)
 ( 2, 6.69713968e-01, 0.52954179, 3.8709697 )
 ( 3, 5.84735305e-01, 0.54795009, 3.34463474)
 ( 4, 4.83432079e-01, 0.57024824, 2.91997823)
 ( 5, 3.77502437e-01, 0.5956425 , 2.59451865)
 ( 6, 2.78389751e-01, 0.6222772 , 2.3546681 )
 ( 7, 1.94497674e-01, 0.64795729, 2.18404118)
 ( 8, 1.29477634e-01, 0.67078251, 2.06671229)
 ( 9, 8.27001827e-02, 0.68957129, 1.98864688)
 (10, 5.10336293e-02, 0.70398283, 1.93831422)
 (11, 3.06148582e-02, 0.7143608 , 1.90680022)
 (12, 1.79468274e-02, 0.7214358 , 1.88758697)
 (13, 1.03240723e-02, 0.7260406 , 1.87614544)
 (14, 5.84768757e-03, 0.72892454, 1.86946918)
 (15, 3.35778195e-03, 0.73067466, 1.8656404 )
 (16, 1.98478537e-03, 0.7317098 , 1.86347648)
 (17, 1.14937826e-03, 0.73230941, 1.86226842)
 (18, 6.54505913e-04, 0.73265086, 1.86160094)
 (19, 3.67583154e-04, 0.73284262, 1.86123535)
 (20, 2.04087865e-04, 0.73294908, 1.86103659)
 (21, 1.12232060e-04, 0.73300761, 1.86092922)
 (22, 6.12220725e-05, 0.73303954, 1.86087154)]
```

Tests
-----

clone repo:

```bash
git clone https://github.com/gmagno/w4
cd w4
```

create virtual environment and install dependencies

```python
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements_dev.txt
```

run tests:

```bash
make test
```

from itertools import count, islice
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from ._cube_lattice import cube_lattice


def int_sin_m(x: float, m: int) -> float:
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    # NOTE for large values of m, numeric integration may be faster than this
    # recursive method
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m


def inverse_increasing(
        func: Callable[[float], float],
        target: float,
        lower: float,
        upper: float,
        atol: float = 1e-10,
) -> float:
    """Returns func inverse of target between lower and upper

    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid


def cube_to_sphere(cube: Sequence[Sequence[float]]) -> Iterator[Tuple[float, ...]]:
    """Map points from [0, 1]^dim to the sphere

    Maps points in [0, 1]^dim to the surface of the sphere; dim + 1 dimensional
    points with unit l2 norms. This mapping preserves relative distance between
    points.

    Parameters
    ----------
    cube : a sequence points in the [0, 1]^dim hyper cube
    """
    dims = {len(p) + 1 for p in cube}
    assert len(dims) == 1, "not all points had the same dimension"
    (dim,) = dims

    output = [[1.0 for _ in range(dim)] for _ in cube]
    mults = [gamma(d / 2 + 0.5) / gamma(d / 2) / sqrt(pi) for d in range(2, dim)]
    for base in cube:
        points = [1.0 for _ in range(dim)]
        points[0] *= sin(2 * pi * base[0])
        points[1] *= cos(2 * pi * base[0])

        for d, (mult, lat) in enumerate(zip(mults, base[1:]), 2):
            deg = inverse_increasing(lambda y: mult * int_sin_m(y, d - 1), lat, 0, pi)
            for j in range(d):
                points[j] *= sin(deg)
            points[d] *= cos(deg)
        yield tuple(points)


def sphere_lattice(dim: int, num_points: int, ) -> List[Tuple[float, ...]]:
    """Generate num_points points over the dim - 1 dimensional hypersphere

    Generate a `num_points` length list of `dim`-dimensional tuples such the
    each element has an l2 norm of 1, and their nearest neighbor is roughly
    identical for each point.

    Parameters
    ----------
    dim : the dimension of points to sample, i.e. the length of tuples in the
        returned list
    num_points : the number of points to generate
    """
    assert dim > 1
    assert num_points > 0
    return list(cube_to_sphere(cube_lattice(dim - 1, num_points)))

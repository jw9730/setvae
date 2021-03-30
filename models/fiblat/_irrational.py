from itertools import count
from math import sqrt
from typing import Iterator


def primes() -> Iterator[int]:
    """Returns an infinite generator of prime numbers

    O(n^{3/2}) for taking n elements
    """
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


def root_primes() -> Iterator[float]:
    """Returns an infinite iterator or mutually irrational numbers

    The numbers are defined as the square roots of successive prime numbers.
    """
    return (sqrt(p) for p in primes())
